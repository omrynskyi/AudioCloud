//! Scratch: applies migrations to a data-directory copy and reports what search now finds.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use audiobank_lib::db::{queries, search, Database};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: <data dir copy>");
    let db = Database::open(&dir, audiobank_lib::EMBEDDING_DIM).expect("open");
    let conn = db.read().unwrap();

    for typed in [
        "kicks", "kick", "808", "snares", "underground", "kic",
        "@prodby.xero", "crown.wav", "Snare - Crown", "808 NOT snare",
        "kick NOT", "\"kick\"", "---", "@prodby.xero Snare - Crown.wav",
    ] {
        let built = search::fts_query(typed);
        let n = queries::search_samples(&conn, typed, 10_000)
            .map(|v| v.len().to_string())
            .unwrap_or_else(|e| format!("ERROR: {e}"));
        println!("{typed:34} -> {n:>10}   [{}]", built.unwrap_or_else(|| "(no constraint)".into()));
    }
}
