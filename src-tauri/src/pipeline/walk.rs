//! Filesystem discovery: `ignore::WalkBuilder::build_parallel()` plus `blake3` hashing.
//!
//! Deliberately **not** `jwalk`, which is deprecated upstream (`overview.md` §3.1).
//! A `(path, mtime, size)` fast-skip runs against existing rows before any hashing, so a
//! re-scan of an unchanged tree never reads file contents. Files over 64 MB are hashed by
//! head+tail+length sampling rather than in full.

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use crossbeam_channel::Sender;
use ignore::{WalkBuilder, WalkState};

use super::{CancellationToken, ScanProgress};
use crate::db::queries::SampleStamp;

/// Extensions the decoder is compiled to handle (`Cargo.toml`'s `symphonia` feature set).
///
/// An allowlist, not a denylist: a sample library is full of `.asd` analysis files, `.als`
/// project files, PDFs and artwork, and probing each one to discover it is not audio costs
/// an open, a read, and a failed probe per file. Lowercase, no leading dot.
///
/// `.opus` is absent on purpose -- `symphonia` has no Opus decoder, so accepting the
/// extension would only manufacture `decode_failed` rows.
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "wave", "bwf", // RIFF
    "aif", "aiff", "aifc", // AIFF
    "caf",  // Core Audio Format -- Logic and Live libraries are full of these
    "flac", // native and Ogg-encapsulated
    "mp3", "m4a", "mp4", "aac", "alac", // MPEG family
    "ogg", "oga", // Ogg/Vorbis
    "mka", // Matroska audio
];

/// Smallest plausible audio file. A canonical WAV header alone is 44 bytes, so anything
/// shorter cannot carry a decodable frame in any container we support. Rejecting these
/// here is what keeps a directory of 0-byte placeholders out of the quarantine list.
pub const MIN_FILE_BYTES: u64 = 44;

/// Sanity ceiling. Above this the file is a disk image or a video that happens to carry an
/// audio extension, not a sample; several container formats cannot address past 4 GiB
/// anyway.
pub const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Above this size, hash by sampling rather than reading the whole file (`overview.md`
/// §3.1). Full-hashing a 2 GB stem to decide whether to embed its first 10 seconds is
/// waste.
pub const HASH_SAMPLE_THRESHOLD: u64 = 64 * 1024 * 1024;

/// How much of the head and of the tail a sampled hash covers.
pub const HASH_SAMPLE_SPAN: u64 = 1024 * 1024;

/// Domain separator folded into a sampled hash.
///
/// Without it, a 1 MiB file and the first mebibyte of a 2 GB file could hash equal and the
/// large file would be deduplicated against the small one. The separator and the length
/// make the two spaces disjoint.
const SAMPLED_HASH_DOMAIN: &[u8] = b"audiobank/blake3-sampled/v1";

/// Read buffer for full hashing. Large enough to amortize the syscall, small enough that
/// one per decode worker is not a memory event.
const HASH_CHUNK_BYTES: usize = 256 * 1024;

/// One audio file the walker decided is worth reading.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub root_id: i64,
    /// Absolute path, as opened.
    pub path: PathBuf,
    /// Path relative to the root, so a root can be moved without invalidating its samples.
    pub rel_path: String,
    pub filename: String,
    /// Lowercased, no leading dot.
    pub ext: String,
    pub size_bytes: i64,
    /// Modification time in milliseconds since the Unix epoch.
    pub mtime: i64,
    /// The row this file already has, if any. Present when the file changed since the last
    /// scan; absent when it is new.
    pub existing: Option<SampleStamp>,
}

/// Everything the walker needs that is not the tree itself.
///
/// Bundled into a struct because `build_parallel`'s visitor closure has to capture all of
/// it by reference and a seven-argument function signature threaded through two closures
/// stops being readable.
#[derive(Debug)]
pub struct WalkContext<'a> {
    pub root_id: i64,
    pub root: &'a Path,
    /// Every `(rel_path -> stamp)` this root already has, loaded in one query before the
    /// walk starts.
    ///
    /// One indexed query beats 50,000 of them, and it takes the read pool out of the hot
    /// path entirely: with four connections and a walker thread per core, per-entry
    /// lookups would spend most of the scan queued on `r2d2`. At 50,000 rows the map is a
    /// few megabytes and it is dropped when the scan ends.
    pub known: &'a HashMap<String, SampleStamp>,
    /// Whether this scan has an inference session behind it, and therefore whether a
    /// `decoded` row counts as finished. See [`crate::db::SampleStatus::is_complete`].
    pub embedding_required: bool,
    pub cancel: &'a CancellationToken,
    pub progress: &'a ScanProgress,
}

/// Walks `root` in parallel, sending every file that needs reading to `sink`.
///
/// Returns when the tree is exhausted or the token is cancelled. Errors on individual
/// entries -- a permission denial, a symlink loop, a directory that vanished mid-walk --
/// are logged and counted, never propagated: one unreadable folder must not abort a scan
/// of a quarter-million files.
pub fn walk_root(ctx: &WalkContext<'_>, sink: &Sender<DiscoveredFile>) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    WalkBuilder::new(ctx.root)
        .threads(threads)
        // Skip dotfiles. This is what keeps `.DS_Store` and `._resource` forks out, and it
        // is the only piece of `ignore`'s filtering we want.
        .hidden(true)
        // Everything below is `ignore`'s VCS-awareness, switched off deliberately. A
        // sample library that happens to live inside a git repository whose `.gitignore`
        // says `*.wav` -- which is a completely ordinary thing for a repository to say --
        // would otherwise scan to zero samples with no error and no explanation. Honoring
        // ignore files is right for a code search tool and wrong for a library indexer.
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .parents(false)
        .require_git(false)
        // Sample libraries genuinely do symlink to external drives, so following links is
        // not optional. `ignore` detects the cycles that creates and reports them as
        // `ErrorKind::Loop`, which the error arm below swallows -- that is the loop guard.
        .follow_links(true)
        .build_parallel()
        .run(|| {
            let sink = sink.clone();
            Box::new(move |entry| visit(ctx, &sink, entry))
        });
}

/// Classifies one directory entry. Runs on every walker thread.
fn visit(
    ctx: &WalkContext<'_>,
    sink: &Sender<DiscoveredFile>,
    entry: Result<ignore::DirEntry, ignore::Error>,
) -> WalkState {
    if ctx.cancel.is_cancelled() {
        return WalkState::Quit;
    }

    let entry = match entry {
        Ok(entry) => entry,
        Err(e) => {
            // `Loop` is the symlink cycle guard firing and is expected on libraries that
            // link to themselves; anything else is a real problem with one entry.
            if is_symlink_loop(&e) {
                tracing::debug!(error = %e, "symlink loop, not descending");
            } else {
                tracing::warn!(error = %e, "skipping unreadable entry");
                ScanProgress::bump(&ctx.progress.failed);
            }
            return WalkState::Continue;
        }
    };

    // `file_type()` is `None` only for the stdin pseudo-entry, which cannot appear here.
    if !entry.file_type().is_some_and(|t| t.is_file()) {
        return WalkState::Continue;
    }
    ScanProgress::bump(&ctx.progress.seen);

    let path = entry.path();
    let Some(ext) = audio_extension(path) else {
        return WalkState::Continue;
    };

    let Ok(meta) = entry.metadata() else {
        tracing::warn!(path = %path.display(), "could not stat, skipping");
        ScanProgress::bump(&ctx.progress.failed);
        return WalkState::Continue;
    };

    let size = meta.len();
    if !(MIN_FILE_BYTES..=MAX_FILE_BYTES).contains(&size) {
        tracing::debug!(path = %path.display(), size, "outside the size bounds");
        return WalkState::Continue;
    }

    // A path that is not UTF-8 cannot round-trip through the `TEXT` column, and a lossy
    // conversion would produce a `rel_path` that no longer opens -- which breaks
    // reveal-in-Finder and every rescan afterwards. macOS normalizes filenames to UTF-8 on
    // both HFS+ and APFS, so this is close to unreachable; it is counted rather than
    // dropped so that "close to" stays visible.
    let Some(rel_path) = relative_path(ctx.root, path) else {
        tracing::warn!(path = %path.display(), "path is not valid UTF-8, skipping");
        ScanProgress::bump(&ctx.progress.failed);
        return WalkState::Continue;
    };

    let mtime = mtime_ms(&meta);

    // The fast skip (`overview.md` §3.1): an unchanged file is never opened, never hashed,
    // and never decoded. This is the whole of the re-scan budget.
    //
    // A row still in `pending` is not skippable however unchanged the file is -- the last
    // scan was interrupted before it produced anything, so "unchanged" describes a file we
    // never actually read. A `decoded` row is skippable only for a scan that is not
    // embedding: with a session available it is a file that still owes a vector, and
    // skipping it is how a resumed scan would leave the corpus permanently half-embedded.
    if let Some(stamp) = ctx.known.get(&rel_path) {
        if stamp.unchanged(mtime, size as i64) && stamp.status.is_complete(ctx.embedding_required) {
            ScanProgress::bump(&ctx.progress.skipped);
            return WalkState::Continue;
        }
    }

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&rel_path)
        .to_string();

    let file = DiscoveredFile {
        root_id: ctx.root_id,
        path: path.to_path_buf(),
        rel_path: rel_path.clone(),
        filename,
        ext,
        size_bytes: size as i64,
        mtime,
        existing: ctx.known.get(&rel_path).copied(),
    };

    ScanProgress::bump(&ctx.progress.queued);

    // A full channel blocks here, and blocking here is the point: backpressure from the
    // decode stage has to reach the walker, or discovery outruns decode by two orders of
    // magnitude and the queue becomes the memory profile (`overview.md` §3).
    if sink.send(file).is_err() {
        // The consumer is gone, which happens only when the scan is tearing down.
        return WalkState::Quit;
    }

    WalkState::Continue
}

/// Whether an entry error is `ignore`'s symlink cycle detection firing.
///
/// `ignore::Error::Loop` is what makes `follow_links(true)` safe, but the walker wraps it
/// in `WithPath`/`WithDepth` before it reaches the visitor, and the crate exposes no
/// accessor that peels those off -- hence the recursion.
fn is_symlink_loop(error: &ignore::Error) -> bool {
    match error {
        ignore::Error::Loop { .. } => true,
        ignore::Error::WithPath { err, .. }
        | ignore::Error::WithDepth { err, .. }
        | ignore::Error::WithLineNumber { err, .. } => is_symlink_loop(err),
        ignore::Error::Partial(errors) => errors.iter().any(is_symlink_loop),
        _ => false,
    }
}

/// The lowercased extension, if it is one we can decode.
fn audio_extension(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    AUDIO_EXTENSIONS.contains(&ext.as_str()).then_some(ext)
}

/// `path` relative to `root`, as UTF-8 with forward slashes.
fn relative_path(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root).ok()?.to_str().map(str::to_string)
}

/// Modification time in milliseconds since the epoch, or 0 for a filesystem that does not
/// record one. Zero is a fine sentinel: it compares unequal to any real stamp, so such a
/// file is simply never fast-skipped.
fn mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A reusable read buffer, so hashing a library does not allocate 256 KiB per file.
///
/// Held by each decode worker for the life of the scan (`overview.md` §3.6).
#[derive(Debug)]
pub struct Hasher {
    buf: Vec<u8>,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            buf: vec![0u8; HASH_CHUNK_BYTES],
        }
    }

    /// Content hash of `path`, in full below [`HASH_SAMPLE_THRESHOLD`] and by head+tail+
    /// length sampling above it.
    ///
    /// `blake3` rather than SHA-2 because this runs over multi-gigabyte scans routinely and
    /// the requirement is collision resistance, not cryptographic strength
    /// (`overview.md` §3.1).
    pub fn hash_file(&mut self, path: &Path, size: u64) -> std::io::Result<[u8; 32]> {
        let mut file = File::open(path)?;
        let mut hasher = blake3::Hasher::new();

        if size <= HASH_SAMPLE_THRESHOLD {
            loop {
                let n = file.read(&mut self.buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&self.buf[..n]);
            }
        } else {
            use std::io::{Seek, SeekFrom};

            hasher.update(SAMPLED_HASH_DOMAIN);
            hasher.update(&size.to_le_bytes());

            read_span(&mut file, HASH_SAMPLE_SPAN, &mut self.buf, &mut hasher)?;
            file.seek(SeekFrom::End(-(HASH_SAMPLE_SPAN as i64)))?;
            read_span(&mut file, HASH_SAMPLE_SPAN, &mut self.buf, &mut hasher)?;
        }

        Ok(*hasher.finalize().as_bytes())
    }
}

/// Feeds up to `span` bytes from the current file position into the hasher, through
/// `scratch`, stopping early at end of file.
fn read_span(
    file: &mut File,
    span: u64,
    scratch: &mut [u8],
    hasher: &mut blake3::Hasher,
) -> std::io::Result<()> {
    let mut remaining = span;
    while remaining > 0 {
        let want = remaining.min(scratch.len() as u64) as usize;
        let n = file.read(&mut scratch[..want])?;
        if n == 0 {
            break;
        }
        hasher.update(&scratch[..n]);
        remaining -= n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::io::Write;

    #[test]
    fn the_extension_filter_is_case_insensitive_and_exclusive() {
        assert_eq!(
            audio_extension(Path::new("/x/KICK.WAV")),
            Some("wav".to_string())
        );
        assert_eq!(
            audio_extension(Path::new("/x/loop.Flac")),
            Some("flac".to_string())
        );
        assert_eq!(audio_extension(Path::new("/x/project.als")), None);
        assert_eq!(audio_extension(Path::new("/x/notes")), None);
        // No Opus decoder is compiled in, so the extension must not be accepted.
        assert_eq!(audio_extension(Path::new("/x/voice.opus")), None);
    }

    #[test]
    fn relative_paths_are_root_relative() {
        let root = Path::new("/Library/Samples");
        assert_eq!(
            relative_path(root, Path::new("/Library/Samples/drums/kick.wav")),
            Some("drums/kick.wav".to_string())
        );
    }

    /// The sampling threshold is the interesting boundary: below it the hash covers every
    /// byte, above it two files that differ only in the middle must still hash apart --
    /// which they do, because the length is folded in but not the middle. This test pins
    /// the property that actually matters: identical content hashes identically, and
    /// different content of the same length does not.
    #[test]
    fn full_hashing_distinguishes_content() {
        let dir = tempfile::tempdir().unwrap();
        let mut hasher = Hasher::new();

        let a = dir.path().join("a.wav");
        let b = dir.path().join("b.wav");
        let c = dir.path().join("c.wav");
        std::fs::write(&a, vec![7u8; 100_000]).unwrap();
        std::fs::write(&b, vec![7u8; 100_000]).unwrap();
        std::fs::write(&c, vec![8u8; 100_000]).unwrap();

        let ha = hasher.hash_file(&a, 100_000).unwrap();
        let hb = hasher.hash_file(&b, 100_000).unwrap();
        let hc = hasher.hash_file(&c, 100_000).unwrap();

        assert_eq!(ha, hb, "identical files must hash identically");
        assert_ne!(ha, hc);
    }

    /// A sampled hash must not collide with the full hash of a file that happens to share
    /// its first mebibyte -- the domain separator and the length are what prevent that.
    #[test]
    fn sampled_hashing_is_domain_separated_from_full_hashing() {
        let dir = tempfile::tempdir().unwrap();
        let mut hasher = Hasher::new();
        let size = HASH_SAMPLE_THRESHOLD + HASH_SAMPLE_SPAN;

        let big = dir.path().join("stem.wav");
        let mut f = File::create(&big).unwrap();
        let chunk = vec![3u8; 1 << 20];
        let mut written = 0u64;
        while written < size {
            let n = chunk.len().min((size - written) as usize);
            f.write_all(&chunk[..n]).unwrap();
            written += n as u64;
        }
        f.sync_all().unwrap();
        drop(f);

        let sampled = hasher.hash_file(&big, size).unwrap();

        // The same bytes, hashed in full, must land somewhere else entirely.
        let full = *blake3::Hasher::new()
            .update(&vec![3u8; size as usize])
            .finalize()
            .as_bytes();
        assert_ne!(sampled, full);

        // ...and sampling is deterministic.
        assert_eq!(sampled, hasher.hash_file(&big, size).unwrap());
    }

    /// Changing only the head, only the tail, or only the length of a large file must all
    /// change its sampled hash. Changing only the middle is the documented blind spot.
    #[test]
    fn sampled_hashing_covers_head_tail_and_length() {
        let dir = tempfile::tempdir().unwrap();
        let mut hasher = Hasher::new();
        let size = HASH_SAMPLE_THRESHOLD + 4096;

        let make = |name: &str, mutate: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = vec![3u8; size as usize];
            mutate(&mut bytes);
            let p = dir.path().join(name);
            std::fs::write(&p, &bytes).unwrap();
            p
        };

        let base = make("base.wav", &|_| {});
        let head = make("head.wav", &|b| b[0] = 9);
        let tail = make("tail.wav", &|b| {
            let last = b.len() - 1;
            b[last] = 9;
        });

        let h_base = hasher.hash_file(&base, size).unwrap();
        assert_ne!(h_base, hasher.hash_file(&head, size).unwrap());
        assert_ne!(h_base, hasher.hash_file(&tail, size).unwrap());
    }
}
