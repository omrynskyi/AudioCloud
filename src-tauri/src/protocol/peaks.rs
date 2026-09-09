//! The `abpeaks://` scheme: serves cached waveform peak summaries with immutable cache
//! headers, keyed by sample id (`overview.md` §6.4).
//!
//! **Transport 3 exists so bulk assets do not compete with commands.** Peaks are per-sample,
//! requested on selection, and pure binary. Routed through `invoke` they would queue behind
//! -- and delay -- the point cloud fetch, the filter query and every other command on the
//! same handler, for a payload the WebView is perfectly capable of caching itself. A
//! registered scheme gets HTTP caching for free, is naturally streamable, and keeps the
//! command channel for commands. The same mechanism serves any future spectrogram image.
//!
//! ### The URL
//!
//! ```text
//! abpeaks://localhost/<sample_id>[?v=<updated_at>]
//! ```
//!
//! **The query string is not decoration.** `Cache-Control: immutable` tells the WebView it
//! never has to revalidate, which is exactly right for "the waveform of sample 1234 as it
//! was at revision N" and exactly wrong for "the waveform of sample 1234", because a
//! rescanned file is different audio under the same id. The full URL is the cache key, so
//! putting the sample's `updated_at` in it makes the immutable promise true. The frontend
//! gets that value from `SampleDetail` and appends it; `src/ipc/peaks.ts` is the only place
//! that URL is built. A request without the parameter still works -- it just shares a cache
//! entry with every other revision of the same sample, which is the behaviour the parameter
//! exists to avoid.

use std::sync::Arc;

use tauri::{
    http::{header, Request, Response, StatusCode},
    Manager, UriSchemeContext,
};

use crate::{audio::peaks::PeakCache, db::Database, error::AppError};

/// The scheme name. Registered on the builder in `lib.rs`.
pub const SCHEME: &str = "abpeaks";

/// A year, which is what `immutable` means in practice.
const CACHE_CONTROL: &str = "max-age=31536000, immutable";

/// Answers one `abpeaks://` request.
///
/// Never panics and never blocks the main thread: Tauri runs a registered scheme handler on
/// a worker, and the only slow thing in here -- decoding a file on a cache miss -- is
/// bounded by one file's first ten seconds.
pub fn handle<R: tauri::Runtime>(
    ctx: UriSchemeContext<'_, R>,
    request: Request<Vec<u8>>,
) -> Response<Vec<u8>> {
    let app = ctx.app_handle();
    let Some(db) = app.try_state::<Database>() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the data layer is not ready",
        );
    };
    let Some(cache) = app.try_state::<PeakCache>() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the peak cache is not ready",
        );
    };

    let Some(sample_id) = parse_id(request.uri().path()) else {
        return error(
            StatusCode::BAD_REQUEST,
            "expected abpeaks://localhost/<sample_id>",
        );
    };

    match cache.get(db.inner(), sample_id) {
        Ok(bytes) => ok(&bytes),
        // 404 rather than 500: from the WebView's point of view a sample that does not
        // exist and a sample whose audio cannot be read are the same thing -- there is no
        // waveform at this URL -- and both are states the inspector already renders from
        // `get_sample_detail`. The typed error is logged, not serialized into the body,
        // because a `fetch` consumer switches on the status and nothing reads an error
        // body it did not ask for.
        Err(AppError::NotFound(what)) => {
            tracing::debug!(sample_id, %what, "abpeaks: no such sample");
            error(StatusCode::NOT_FOUND, "no such sample")
        }
        Err(e) => {
            tracing::warn!(sample_id, error = %e, "abpeaks: could not produce a summary");
            error(StatusCode::NOT_FOUND, "no waveform for this sample")
        }
    }
}

/// Extracts the sample id from `/<id>`.
///
/// Rejects anything that is not a positive integer, which is the whole of the parsing this
/// scheme needs: the id goes to a bound query parameter and never near a path. The frontend
/// has no filesystem permission at all (`overview.md` §2), and this handler is the one place
/// a URL from the WebView reaches the data layer -- so it accepts an integer, or nothing.
fn parse_id(path: &str) -> Option<i64> {
    let id: i64 = path.trim_matches('/').parse().ok()?;
    (id > 0).then_some(id)
}

fn ok(bytes: &Arc<Vec<u8>>) -> Response<Vec<u8>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CACHE_CONTROL, CACHE_CONTROL)
        .header(header::CONTENT_LENGTH, bytes.len())
        // The WebView origin is not this scheme's origin, so without this the `fetch`
        // is blocked before the body is ever read.
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(bytes.as_ref().clone())
        .unwrap_or_else(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "could not build a response"))
}

/// A bodyless failure. `unwrap_or_else` is not reachable -- a builder with a status and no
/// headers cannot fail -- but `clippy::unwrap_used` is a warning in this crate for a reason
/// and a scheme handler has no way to report an error other than by being one.
fn error(status: StatusCode, reason: &'static str) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(reason.as_bytes().to_vec())
        .unwrap_or_else(|_| Response::new(Vec::new()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn a_path_is_a_sample_id_or_nothing() {
        assert_eq!(parse_id("/1234"), Some(1234));
        assert_eq!(parse_id("1234"), Some(1234));
        assert_eq!(parse_id("/1234/"), Some(1234));

        for hostile in [
            "/../../etc/passwd",
            "/1234; DROP TABLE samples",
            "/0",
            "/-1",
            "/",
            "",
            "/1234/extra",
            "/0x10",
        ] {
            assert_eq!(parse_id(hostile), None, "accepted {hostile:?}");
        }
    }
}
