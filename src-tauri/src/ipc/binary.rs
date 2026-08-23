//! Transport 2: raw bytes over `tauri::ipc::Response` (`overview.md` §6.3).
//!
//! Three payloads share one frame. The frame is sixteen bytes -- magic, version, count,
//! reserved -- and every body that follows it is **struct-of-arrays**: all the ids, then all
//! the xs, then all the ys, then all the zs. Not array-of-structs, and not JSON.
//!
//! **The header's size is load-bearing.** `new Float32Array(buffer, offset, n)` throws if
//! `offset` is not a multiple of four, and `new Float64Array` wants eight. Sixteen bytes is
//! a multiple of every alignment a typed array can ask for, so the body can start with any
//! column type and later versions can add one without moving anything. The fourth word is
//! reserved rather than removed for the same reason: dropping it would make the header
//! twelve bytes and every f64 column thereafter unrepresentable.
//!
//! **Struct-of-arrays, for the consumer's sake.** Three's `BufferGeometry` wants interleaved
//! XYZ, so the frontend weaves the planar columns once on arrival -- a few milliseconds over
//! 50,000 points. Interleaving on the wire instead would save that loop and cost the ability
//! to fetch one column on its own, which is exactly what `get_feature_column` does.
//!
//! ### The frames
//!
//! ```text
//! header   [0..4)   magic, four ASCII bytes
//!          [4..8)   version, u32 LE
//!          [8..12)  count, u32 LE
//!          [12..16) reserved, u32 LE -- zero, except in ABPK where it is `covered_ms`
//!
//! ABPC     ids u32[count], xs f32[count], ys f32[count], zs f32[count]
//! ABFC     values f32[count], in the point cloud's order, NaN where the column is null
//! ABQS     ids u32[count], ascending
//! ABPK     min/max f32 pairs, 2*count floats, interleaved
//! ```
//!
//! Everything is little-endian, which is not a portability claim -- it is the only byte
//! order Apple Silicon and `DataView`'s default disagree about, and `DataView` is what the
//! frontend uses for the header. The typed-array views over the body are host-endian by
//! definition, and the host is little-endian on every machine this ships to.

use crate::error::AppError;

/// Wire format version. Bumped when a body layout changes in a way an old decoder would
/// misread; the frontend refuses a version it does not know rather than reading garbage.
pub const VERSION: u32 = 1;

/// Bytes before the first column. See the module note on why it is sixteen and not twelve.
pub const HEADER_BYTES: usize = 16;

/// `get_point_cloud`.
pub const MAGIC_POINT_CLOUD: &[u8; 4] = b"ABPC";
/// `get_feature_column`.
pub const MAGIC_FEATURE_COLUMN: &[u8; 4] = b"ABFC";
/// `query_samples`.
pub const MAGIC_QUERY_RESULT: &[u8; 4] = b"ABQS";
/// The `abpeaks://` scheme's body.
pub const MAGIC_PEAKS: &[u8; 4] = b"ABPK";

/// Starts a frame with `count` elements.
///
/// `reserved` is zero for every payload but `ABPK`, which spends it on the number of
/// milliseconds its buckets actually cover -- see [`peaks`].
fn frame(magic: &[u8; 4], count: usize, reserved: u32, body_bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_BYTES + body_bytes);
    buf.extend_from_slice(magic);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(count as u32).to_le_bytes());
    buf.extend_from_slice(&reserved.to_le_bytes());
    buf
}

/// Narrows a SQLite row id to the `u32` the wire carries.
///
/// Sample ids are `INTEGER PRIMARY KEY`, so SQLite hands out `max(id) + 1` and reuses the
/// space freed by deletions -- an id above four billion means a library that has churned
/// through four billion rows, which is not a thing that happens. It is checked anyway,
/// because the alternative to checking is a silent truncation that puts one point at the
/// wrong coordinate and gives no way to find out.
fn narrow(id: i64) -> Result<u32, AppError> {
    u32::try_from(id).map_err(|_| {
        AppError::internal(
            "narrowing a sample id for the binary transport",
            format!("sample id {id} does not fit in u32"),
        )
    })
}

/// Encodes the point cloud: ids, then xs, then ys, then zs.
///
/// 16 + 16n bytes. At the 50,000 points every §7 target is stated against that is
/// **800,016 bytes**, against a 900 KB budget.
pub fn point_cloud(points: &[(i64, [f32; 3])]) -> Result<Vec<u8>, AppError> {
    let n = points.len();
    let mut buf = frame(MAGIC_POINT_CLOUD, n, 0, n * 16);

    for (id, _) in points {
        buf.extend_from_slice(&narrow(*id)?.to_le_bytes());
    }
    for axis in 0..3 {
        for (_, p) in points {
            buf.extend_from_slice(&p[axis].to_le_bytes());
        }
    }

    Ok(buf)
}

/// Encodes one column of `f32`, in the point cloud's order.
///
/// **Order is the contract**, and it is the only thing tying a value back to a point: the
/// column carries no ids, because 200 KB of ids per column on a payload that is fetched
/// every time the user changes what the map is coloured by is a cost with no buyer. Both
/// this and [`point_cloud`] read the active run through the same `ORDER BY sample_id`, so
/// index `i` is the same sample in both. The `count` in the header is what makes a
/// mismatched pair -- a re-fit landed between the two fetches -- a caught error on the
/// frontend rather than a map coloured by the wrong numbers.
///
/// A null cell is `f32::NAN`. Not zero, and not a parallel presence bitmap: zero is a
/// legitimate value for every column here, NaN is not, and `Number.isNaN` is a cheaper test
/// on the JS side than a bit lookup.
pub fn feature_column(values: &[f32]) -> Vec<u8> {
    let mut buf = frame(MAGIC_FEATURE_COLUMN, values.len(), 0, values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Encodes a filter result: sample ids, ascending.
///
/// Ascending because the point cloud is also in `sample_id` order, so the frontend can turn
/// "which points are in this filter" into one merge over two sorted arrays instead of
/// building a 30,000-entry `Set` on every keystroke.
pub fn id_list(ids: &[i64]) -> Result<Vec<u8>, AppError> {
    let mut buf = frame(MAGIC_QUERY_RESULT, ids.len(), 0, ids.len() * 4);
    for id in ids {
        buf.extend_from_slice(&narrow(*id)?.to_le_bytes());
    }
    Ok(buf)
}

/// Encodes a waveform summary: `count` buckets of interleaved `(min, max)` spanning
/// `covered_ms` of audio.
///
/// Interleaved here, against the struct-of-arrays rule everywhere else, and for the reason
/// the rule exists: the consumer walks buckets, drawing one vertical line per bucket from
/// min to max. Two planar arrays would make that loop stride two cache lines instead of
/// one, and there is no column here anyone fetches independently.
///
/// **`covered_ms` is what stops the waveform from lying.** The decoder stops at
/// `decode::WINDOW_SECONDS`, so the summary of a four-minute loop describes its first ten
/// seconds. Drawn edge to edge under a label reading "4:07" that is simply wrong. The
/// frontend compares this against the sample's real `durationMs` and marks where the
/// summary stops.
pub fn peaks(minmax: &[(f32, f32)], covered_ms: u32) -> Vec<u8> {
    let mut buf = frame(MAGIC_PEAKS, minmax.len(), covered_ms, minmax.len() * 8);
    for (lo, hi) in minmax {
        buf.extend_from_slice(&lo.to_le_bytes());
        buf.extend_from_slice(&hi.to_le_bytes());
    }
    buf
}

/// Reads a frame header back. The decoder half of the format, used by the tests and by
/// anything in-process that wants to assert on a payload.
///
/// The frontend has its own copy of this in `src/ipc/binary.ts`; there is no way to share
/// one implementation across a process boundary, so the two are kept honest by
/// `tests/ipc.rs` asserting the byte offsets this function reads.
pub fn header(buf: &[u8]) -> Option<Header<'_>> {
    if buf.len() < HEADER_BYTES {
        return None;
    }
    Some(Header {
        magic: &buf[0..4],
        version: u32::from_le_bytes(buf[4..8].try_into().ok()?),
        count: u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize,
        reserved: u32::from_le_bytes(buf[12..16].try_into().ok()?),
    })
}

/// A decoded frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header<'a> {
    pub magic: &'a [u8],
    pub version: u32,
    pub count: usize,
    pub reserved: u32,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn cloud(n: usize) -> Vec<(i64, [f32; 3])> {
        (0..n)
            .map(|i| (i as i64 + 1, [i as f32, -(i as f32), 0.5]))
            .collect()
    }

    #[test]
    fn the_header_is_sixteen_bytes_and_says_what_it_is() {
        let buf = point_cloud(&cloud(3)).unwrap();
        let h = header(&buf).unwrap();
        assert_eq!(h.magic, MAGIC_POINT_CLOUD);
        assert_eq!(h.version, VERSION);
        assert_eq!(h.count, 3);
        assert_eq!(h.reserved, 0);
    }

    /// Struct-of-arrays, and the offsets the frontend computes from `count`.
    #[test]
    fn the_point_cloud_body_is_four_planar_columns() {
        let points = cloud(4);
        let buf = point_cloud(&points).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES + 4 * 16);

        let column = |axis: usize| {
            let start = HEADER_BYTES + (1 + axis) * 4 * 4;
            (0..4)
                .map(|i| {
                    let at = start + i * 4;
                    f32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
                })
                .collect::<Vec<_>>()
        };

        let ids: Vec<u32> = (0..4)
            .map(|i| {
                let at = HEADER_BYTES + i * 4;
                u32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(column(0), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(column(1), vec![0.0, -1.0, -2.0, -3.0]);
        assert_eq!(column(2), vec![0.5; 4]);
    }

    /// Every column start must be four-byte aligned or the frontend's `Float32Array` view
    /// throws. With a 16-byte header and 4-byte elements this is arithmetic, not luck --
    /// which is exactly why it is worth an assertion that would fail the day someone adds a
    /// three-byte field to the header.
    #[test]
    fn every_column_offset_is_four_byte_aligned() {
        for n in [0, 1, 3, 1000] {
            let buf = point_cloud(&cloud(n)).unwrap();
            for axis in 0..4 {
                assert_eq!((HEADER_BYTES + axis * n * 4) % 4, 0);
            }
            assert_eq!(buf.len(), HEADER_BYTES + n * 16);
        }
    }

    /// `overview.md` §7: the point cloud payload is budgeted at 900 KB.
    #[test]
    fn fifty_thousand_points_fit_the_budget() {
        let buf = point_cloud(&cloud(50_000)).unwrap();
        assert_eq!(buf.len(), 800_016);
        assert!(
            buf.len() <= 900 * 1024,
            "point cloud payload is {} bytes",
            buf.len()
        );
    }

    /// A null cell has to survive the wire as something the frontend can test for, and
    /// `f32::NAN` does not compare equal to itself -- so the assertion is on the bit
    /// pattern's NaN-ness, not on equality.
    #[test]
    fn a_null_feature_cell_arrives_as_nan() {
        let buf = feature_column(&[1.0, f32::NAN, -3.5]);
        let at = |i: usize| {
            let start = HEADER_BYTES + i * 4;
            f32::from_le_bytes(buf[start..start + 4].try_into().unwrap())
        };
        assert_eq!(at(0), 1.0);
        assert!(at(1).is_nan());
        assert_eq!(at(2), -3.5);
    }

    #[test]
    fn an_id_that_does_not_fit_is_an_error_rather_than_a_truncation() {
        let err = point_cloud(&[(u32::MAX as i64 + 1, [0.0; 3])]).unwrap_err();
        assert!(matches!(err, AppError::Internal(_)));
        assert!(id_list(&[i64::MAX]).is_err());
    }

    #[test]
    fn empty_payloads_are_a_bare_header() {
        for buf in [
            point_cloud(&[]).unwrap(),
            feature_column(&[]),
            id_list(&[]).unwrap(),
            peaks(&[], 0),
        ] {
            assert_eq!(buf.len(), HEADER_BYTES);
            assert_eq!(header(&buf).unwrap().count, 0);
        }
    }

    #[test]
    fn peaks_are_interleaved_min_max_and_say_how_much_audio_they_cover() {
        let buf = peaks(&[(-1.0, 1.0), (-0.25, 0.5)], 10_000);
        assert_eq!(buf.len(), HEADER_BYTES + 16);
        assert_eq!(header(&buf).unwrap().reserved, 10_000);
        let at = |i: usize| {
            let start = HEADER_BYTES + i * 4;
            f32::from_le_bytes(buf[start..start + 4].try_into().unwrap())
        };
        assert_eq!([at(0), at(1), at(2), at(3)], [-1.0, 1.0, -0.25, 0.5]);
    }
}
