//! `embeddings.bin`: the embedding matrix lives outside SQLite (`overview.md` §4.2).
//!
//! Append-only, f16 storage via `half`, addressed by the `(offset, len)` pair recorded on
//! the sample row. Reads `mmap` the whole matrix rather than heap-loading it, so a 50k ×
//! 512 corpus costs page cache instead of 51 MB of resident memory. [`EmbeddingStore::compact`]
//! reclaims space after bulk deletion.
//!
//! **Layout.** A flat run of little-endian f16 values with no header and no padding.
//! `samples.emb_offset` is a byte offset into the file; `samples.emb_len` is the vector's
//! dimensionality, so its byte length is `emb_len * 2`. There is deliberately no index in
//! the file: SQLite already holds it.
//!
//! **Why f16.** 51 MB instead of 102 MB, and the precision loss sits far below the noise
//! floor of a UMAP neighborhood computation. Values widen to f32 on read.

use std::{
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use half::f16;
use memmap2::Mmap;

use super::{io_error, DbError};

/// Filename inside the app data directory.
pub const EMBEDDINGS_FILENAME: &str = "embeddings.bin";

/// Bytes per stored value.
const BYTES_PER_VALUE: u64 = 2;

/// Where one sample's vector lives in `embeddings.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingLoc {
    /// Byte offset from the start of the file.
    pub offset: u64,
    /// Number of dimensions (not bytes). Stored in `samples.emb_len`.
    pub dims: u32,
}

impl EmbeddingLoc {
    /// Length of this vector in bytes.
    pub fn bytes(&self) -> u64 {
        u64::from(self.dims) * BYTES_PER_VALUE
    }
}

/// The append-only vector file.
///
/// Writes go through one owner (`&mut self`), matching the writer-thread discipline of the
/// database itself. Reads are `&self` and can happen concurrently through [`Self::read`]
/// or a [`EmbeddingMatrix`].
#[derive(Debug)]
pub struct EmbeddingStore {
    path: PathBuf,
    file: File,
    len: u64,
    dim: usize,
    /// Reused encode buffer, so appending does not allocate per vector.
    scratch: Vec<u8>,
}

impl EmbeddingStore {
    /// Opens (creating if absent) `embeddings.bin` under `dir`, expecting `dim`-dimensional
    /// vectors.
    pub fn open(dir: &Path, dim: usize) -> Result<Self, DbError> {
        let path = dir.join(EMBEDDINGS_FILENAME);
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|e| io_error(format!("opening {}", path.display()), e))?;

        let len = file
            .metadata()
            .map_err(|e| io_error(format!("stat {}", path.display()), e))?
            .len();

        Ok(Self {
            path,
            file,
            len,
            dim,
            scratch: Vec::new(),
        })
    }

    /// Dimensionality this store accepts.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Current size of the file in bytes.
    pub fn len_bytes(&self) -> u64 {
        self.len
    }

    /// Path to `embeddings.bin`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one vector and returns where it landed.
    pub fn append(&mut self, vector: &[f32]) -> Result<EmbeddingLoc, DbError> {
        self.append_batch(std::slice::from_ref(&vector))
            .map(|mut locs| locs.remove(0))
    }

    /// Appends many vectors in one write.
    ///
    /// One `write_all` for the whole batch: at 50,000 vectors the difference between this
    /// and a syscall per vector is most of the wall clock.
    pub fn append_batch<V: AsRef<[f32]>>(
        &mut self,
        vectors: &[V],
    ) -> Result<Vec<EmbeddingLoc>, DbError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        self.scratch.clear();
        self.scratch
            .reserve(vectors.len() * self.dim * BYTES_PER_VALUE as usize);

        let mut locs = Vec::with_capacity(vectors.len());
        let mut offset = self.len;

        for vector in vectors {
            let vector = vector.as_ref();
            if vector.len() != self.dim {
                return Err(DbError::Dimension {
                    expected: self.dim,
                    actual: vector.len(),
                });
            }
            for value in vector {
                self.scratch
                    .extend_from_slice(&f16::from_f32(*value).to_le_bytes());
            }
            let loc = EmbeddingLoc {
                offset,
                dims: self.dim as u32,
            };
            offset += loc.bytes();
            locs.push(loc);
        }

        self.file
            .write_all(&self.scratch)
            .map_err(|e| io_error(format!("appending to {}", self.path.display()), e))?;
        self.len = offset;

        Ok(locs)
    }

    /// Flushes the file to disk. Called at the end of a scan, not per append.
    pub fn sync(&self) -> Result<(), DbError> {
        self.file
            .sync_data()
            .map_err(|e| io_error(format!("syncing {}", self.path.display()), e))
    }

    /// Reads one vector, widened to f32.
    ///
    /// This is the nearest-neighbor path: one row, by offset. Whole-matrix work uses
    /// [`Self::matrix`] instead.
    pub fn read(&self, loc: EmbeddingLoc) -> Result<Vec<f32>, DbError> {
        self.check_bounds(loc, self.len)?;

        let mut bytes = vec![0u8; loc.bytes() as usize];
        self.file
            .read_exact_at(&mut bytes, loc.offset)
            .map_err(|e| io_error(format!("reading {}", self.path.display()), e))?;

        Ok(widen(&bytes))
    }

    /// Memory-maps the whole file for projection work.
    ///
    /// The map is a view, not a load: pages arrive as they are touched and the kernel can
    /// evict them under pressure. This is what lets a 50k × 512 re-fit run without a 102 MB
    /// heap allocation.
    pub fn matrix(&self) -> Result<EmbeddingMatrix, DbError> {
        if self.len == 0 {
            return Ok(EmbeddingMatrix { map: None, len: 0 });
        }

        // SAFETY: mapping a file is unsound in general because another process can truncate
        // it under the mapping, turning reads into SIGBUS. `embeddings.bin` lives in the
        // app's own data directory, is append-only, and is only ever truncated by
        // `compact()`, which writes a fresh file and renames over this one -- a rename
        // leaves the existing mapping pointing at the old, still-intact inode.
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&self.file) }
            .map_err(|e| io_error(format!("mapping {}", self.path.display()), e))?;

        Ok(EmbeddingMatrix {
            len: map.len() as u64,
            map: Some(map),
        })
    }

    /// Fraction of the file no live sample points at.
    ///
    /// Phase 10 triggers [`Self::compact`] above ~25%.
    pub fn dead_ratio(&self, live_bytes: u64) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        1.0 - (live_bytes.min(self.len) as f64 / self.len as f64)
    }

    /// Rewrites the file containing only `live`, in the order given.
    ///
    /// Returns each entry's new location. The caller must persist those offsets inside one
    /// transaction; until it does, the database points into a file that no longer matches.
    /// Bytes are copied verbatim rather than decoded and re-encoded, so compaction is
    /// bit-exact.
    pub fn compact(
        &mut self,
        live: &[(i64, EmbeddingLoc)],
    ) -> Result<Vec<(i64, EmbeddingLoc)>, DbError> {
        let tmp_path = self.path.with_extension("bin.compacting");
        let mut tmp = File::create(&tmp_path)
            .map_err(|e| io_error(format!("creating {}", tmp_path.display()), e))?;

        let mut moved = Vec::with_capacity(live.len());
        let mut offset = 0u64;
        let mut buf = Vec::new();

        for (sample_id, loc) in live {
            self.check_bounds(*loc, self.len)?;
            buf.resize(loc.bytes() as usize, 0);
            self.file
                .read_exact_at(&mut buf, loc.offset)
                .map_err(|e| io_error(format!("reading {}", self.path.display()), e))?;
            tmp.write_all(&buf)
                .map_err(|e| io_error(format!("writing {}", tmp_path.display()), e))?;

            moved.push((
                *sample_id,
                EmbeddingLoc {
                    offset,
                    dims: loc.dims,
                },
            ));
            offset += loc.bytes();
        }

        // fsync before rename: a rename is atomic with respect to *ordering*, not with
        // respect to *durability*. Without this the directory entry can reach disk ahead
        // of the contents it points at.
        tmp.sync_all()
            .map_err(|e| io_error(format!("syncing {}", tmp_path.display()), e))?;
        drop(tmp);

        std::fs::rename(&tmp_path, &self.path).map_err(|e| {
            io_error(
                format!(
                    "replacing {} with {}",
                    self.path.display(),
                    tmp_path.display()
                ),
                e,
            )
        })?;

        // The old handle still refers to the replaced inode; reopen onto the new file.
        let store = Self::open(
            self.path.parent().unwrap_or_else(|| Path::new(".")),
            self.dim,
        )?;
        self.file = store.file;
        self.len = store.len;

        Ok(moved)
    }

    fn check_bounds(&self, loc: EmbeddingLoc, size: u64) -> Result<(), DbError> {
        let end = loc.offset.saturating_add(loc.bytes());
        if end > size {
            return Err(DbError::EmbeddingOutOfRange {
                offset: loc.offset,
                bytes: loc.bytes(),
                size,
            });
        }
        Ok(())
    }
}

/// A memory-mapped view of the whole embedding file.
///
/// Rows are widened one at a time into a caller-owned buffer. Nothing here ever
/// materializes the full f32 matrix -- that is the entire point of §4.2.
#[derive(Debug)]
pub struct EmbeddingMatrix {
    map: Option<Mmap>,
    len: u64,
}

impl EmbeddingMatrix {
    /// Mapped size in bytes.
    pub fn len_bytes(&self) -> u64 {
        self.len
    }

    /// Whether the file was empty when mapped.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Widens one row into `out`, reusing its allocation.
    pub fn row_into(&self, loc: EmbeddingLoc, out: &mut Vec<f32>) -> Result<(), DbError> {
        let end = loc.offset.saturating_add(loc.bytes());
        let bytes = match &self.map {
            Some(map) if end <= self.len => &map[loc.offset as usize..end as usize],
            _ => {
                return Err(DbError::EmbeddingOutOfRange {
                    offset: loc.offset,
                    bytes: loc.bytes(),
                    size: self.len,
                })
            }
        };

        out.clear();
        out.reserve(loc.dims as usize);
        out.extend(
            bytes
                .chunks_exact(BYTES_PER_VALUE as usize)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()),
        );
        Ok(())
    }

    /// Widens one row into a fresh `Vec`.
    pub fn row(&self, loc: EmbeddingLoc) -> Result<Vec<f32>, DbError> {
        let mut out = Vec::new();
        self.row_into(loc, &mut out)?;
        Ok(out)
    }
}

/// Widens a run of little-endian f16 bytes to f32.
///
/// `chunks_exact` rather than a pointer cast: the f16 values are copied out either way
/// because the caller wants f32, so a zero-copy `&[f16]` view would buy nothing and cost
/// an alignment assumption.
fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(BYTES_PER_VALUE as usize)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    const DIM: usize = 8;

    fn store() -> (tempfile::TempDir, EmbeddingStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = EmbeddingStore::open(dir.path(), DIM).unwrap();
        (dir, store)
    }

    /// A vector as it survives the f16 narrowing. Every round-trip assertion compares
    /// against this, not against the f32 input: the storage format is lossy by design and
    /// pretending otherwise would test the wrong thing.
    fn as_stored(v: &[f32]) -> Vec<f32> {
        v.iter().map(|x| f16::from_f32(*x).to_f32()).collect()
    }

    fn vector(seed: usize) -> Vec<f32> {
        (0..DIM)
            .map(|i| ((seed * DIM + i) as f32 / 97.0) - 0.5)
            .collect()
    }

    #[test]
    fn a_vector_round_trips_through_the_file() {
        let (_dir, mut store) = store();
        let v = vector(3);

        let loc = store.append(&v).unwrap();

        assert_eq!(loc.offset, 0);
        assert_eq!(loc.dims, DIM as u32);
        assert_eq!(loc.bytes(), DIM as u64 * 2);
        assert_eq!(store.read(loc).unwrap(), as_stored(&v));
    }

    #[test]
    fn appends_are_placed_end_to_end() {
        let (_dir, mut store) = store();
        let vectors: Vec<Vec<f32>> = (0..4).map(vector).collect();

        let locs = store.append_batch(&vectors).unwrap();

        for (i, loc) in locs.iter().enumerate() {
            assert_eq!(loc.offset, (i * DIM * 2) as u64);
            assert_eq!(store.read(*loc).unwrap(), as_stored(&vectors[i]));
        }
        assert_eq!(store.len_bytes(), (4 * DIM * 2) as u64);
    }

    /// The mmap path and the pread path must agree value for value -- the projection reads
    /// through one and the inspector through the other.
    #[test]
    fn the_mapped_matrix_agrees_with_a_direct_read() {
        let (_dir, mut store) = store();
        let vectors: Vec<Vec<f32>> = (0..64).map(vector).collect();
        let locs = store.append_batch(&vectors).unwrap();

        let matrix = store.matrix().unwrap();
        assert_eq!(matrix.len_bytes(), store.len_bytes());

        let mut row = Vec::new();
        for (i, loc) in locs.iter().enumerate() {
            matrix.row_into(*loc, &mut row).unwrap();
            assert_eq!(row, as_stored(&vectors[i]));
            assert_eq!(row, store.read(*loc).unwrap());
        }
    }

    #[test]
    fn a_wrong_length_vector_is_rejected_rather_than_stored() {
        let (_dir, mut store) = store();

        let err = store.append(&[0.0; DIM + 1]).unwrap_err();

        assert!(matches!(
            err,
            DbError::Dimension {
                expected: DIM,
                actual: 9
            }
        ));
        assert_eq!(store.len_bytes(), 0, "a rejected append must write nothing");
    }

    /// A stale `(offset, len)` -- from a compaction the database never learned about --
    /// must surface as a typed error, not as garbage floats or a panic.
    #[test]
    fn a_location_past_the_end_of_the_file_is_an_error() {
        let (_dir, mut store) = store();
        store.append(&vector(0)).unwrap();

        let bogus = EmbeddingLoc {
            offset: 4096,
            dims: DIM as u32,
        };
        assert!(matches!(
            store.read(bogus),
            Err(DbError::EmbeddingOutOfRange { .. })
        ));
        assert!(matches!(
            store.matrix().unwrap().row(bogus),
            Err(DbError::EmbeddingOutOfRange { .. })
        ));
    }

    #[test]
    fn an_empty_store_maps_without_error() {
        let (_dir, store) = store();
        let matrix = store.matrix().unwrap();
        assert!(matrix.is_empty());
        assert_eq!(store.dead_ratio(0), 0.0);
    }

    #[test]
    fn compaction_drops_the_holes_and_keeps_the_bytes_identical() {
        let (_dir, mut store) = store();
        let vectors: Vec<Vec<f32>> = (0..6).map(vector).collect();
        let locs = store.append_batch(&vectors).unwrap();

        // Samples 1, 3, and 5 were deleted; the even ones survive.
        let live: Vec<(i64, EmbeddingLoc)> = [0usize, 2, 4]
            .iter()
            .map(|&i| (i as i64 + 100, locs[i]))
            .collect();
        assert!((store.dead_ratio(3 * DIM as u64 * 2) - 0.5).abs() < f64::EPSILON);

        let moved = store.compact(&live).unwrap();

        assert_eq!(store.len_bytes(), (3 * DIM * 2) as u64);
        for (n, (sample_id, loc)) in moved.iter().enumerate() {
            assert_eq!(*sample_id, live[n].0);
            assert_eq!(loc.offset, (n * DIM * 2) as u64);
            assert_eq!(
                store.read(*loc).unwrap(),
                as_stored(&vectors[n * 2]),
                "compaction must move bytes verbatim, not re-encode them"
            );
        }
        assert_eq!(store.dead_ratio(3 * DIM as u64 * 2), 0.0);
    }

    /// Compaction reopens the file; appending afterwards must continue from the new end,
    /// not from where the old handle happened to be.
    #[test]
    fn appending_after_compaction_continues_from_the_new_end() {
        let (_dir, mut store) = store();
        let locs = store
            .append_batch(&(0..3).map(vector).collect::<Vec<_>>())
            .unwrap();

        let moved = store.compact(&[(1, locs[2])]).unwrap();
        let added = store.append(&vector(9)).unwrap();

        assert_eq!(moved[0].1.offset, 0);
        assert_eq!(added.offset, (DIM * 2) as u64);
        assert_eq!(store.read(added).unwrap(), as_stored(&vector(9)));
        assert_eq!(store.len_bytes(), (2 * DIM * 2) as u64);
    }

    #[test]
    fn a_reopened_store_appends_after_what_is_already_there() {
        let dir = tempfile::tempdir().unwrap();

        let first = {
            let mut store = EmbeddingStore::open(dir.path(), DIM).unwrap();
            let loc = store.append(&vector(1)).unwrap();
            store.sync().unwrap();
            loc
        };

        let mut reopened = EmbeddingStore::open(dir.path(), DIM).unwrap();
        let second = reopened.append(&vector(2)).unwrap();

        assert_eq!(first.offset, 0);
        assert_eq!(second.offset, (DIM * 2) as u64);
        assert_eq!(reopened.read(first).unwrap(), as_stored(&vector(1)));
    }
}
