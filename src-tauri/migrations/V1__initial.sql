-- V1: the initial schema (overview.md 4.1).
--
-- Forward-only. The tables that hold real user work -- library_roots, tags, collections,
-- and their join tables -- are the ones a pre-migration backup exists to protect
-- (overview.md 4.5). Everything else is a derived cache that a rescan can rebuild.

-- Library roots the user has added. Kept separate so a root can be
-- rescanned, disabled, or removed without orphaning sample rows.
CREATE TABLE library_roots (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL UNIQUE,
    label         TEXT,
    enabled       INTEGER NOT NULL DEFAULT 1,
    added_at      INTEGER NOT NULL,
    last_scan_id  INTEGER REFERENCES scan_runs(id) ON DELETE SET NULL
);

CREATE TABLE samples (
    id            INTEGER PRIMARY KEY,
    root_id       INTEGER NOT NULL REFERENCES library_roots(id) ON DELETE CASCADE,
    rel_path      TEXT    NOT NULL,          -- relative to root, so roots can move
    filename      TEXT    NOT NULL,
    ext           TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    mtime         INTEGER NOT NULL,          -- with size, the cheap "unchanged?" check
    content_hash  BLOB,                      -- blake3, 32 bytes; NULL until hashed
    duration_ms   INTEGER,
    sample_rate   INTEGER,
    channels      INTEGER,
    status        TEXT    NOT NULL DEFAULT 'pending',
                  -- pending | decoded | embedded | decode_failed | missing
    error         TEXT,
    -- Location in embeddings.bin. NULL until the embed stage completes.
    emb_offset    INTEGER,
    emb_len       INTEGER,
    first_seen_at INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    UNIQUE (root_id, rel_path)
);

CREATE INDEX idx_samples_status ON samples(status);
CREATE INDEX idx_samples_hash   ON samples(content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX idx_samples_root   ON samples(root_id);

-- Separate table: written by a different pipeline stage, queried by the
-- filter UI far more often than the sample row, and nullable as a block.
CREATE TABLE sample_features (
    sample_id         INTEGER PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
    peak_db           REAL,
    rms_db            REAL,
    lufs_integrated   REAL,
    spectral_centroid REAL,
    spectral_flatness REAL,
    zero_crossing     REAL,
    onset_density     REAL,
    bpm               REAL,
    bpm_confidence    REAL,
    key_root          INTEGER,   -- 0-11, NULL if unpitched
    key_mode          INTEGER,   -- 0 minor, 1 major
    key_confidence    REAL
);

CREATE INDEX idx_features_centroid ON sample_features(spectral_centroid);
CREATE INDEX idx_features_bpm      ON sample_features(bpm);

-- Versioned so a re-fit can be built in the background and swapped atomically.
CREATE TABLE projection_runs (
    id           INTEGER PRIMARY KEY,
    algorithm    TEXT    NOT NULL,           -- 'umap' | 'pca'
    params_json  TEXT    NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at   INTEGER NOT NULL,
    completed_at INTEGER,
    is_active    INTEGER NOT NULL DEFAULT 0
);

-- Exactly one active projection at a time. Enforced, not just intended.
CREATE UNIQUE INDEX idx_projection_active
    ON projection_runs(is_active) WHERE is_active = 1;

CREATE TABLE projections (
    run_id    INTEGER NOT NULL REFERENCES projection_runs(id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    PRIMARY KEY (run_id, sample_id)
) WITHOUT ROWID;

CREATE TABLE tags (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE,
    color TEXT
);

CREATE TABLE sample_tags (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    tag_id    INTEGER NOT NULL REFERENCES tags(id)    ON DELETE CASCADE,
    PRIMARY KEY (sample_id, tag_id)
) WITHOUT ROWID;

CREATE INDEX idx_sample_tags_tag ON sample_tags(tag_id);

CREATE TABLE collections (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE collection_members (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    sample_id     INTEGER NOT NULL REFERENCES samples(id)     ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    PRIMARY KEY (collection_id, sample_id)
) WITHOUT ROWID;

CREATE TABLE scan_runs (
    id             INTEGER PRIMARY KEY,
    root_id        INTEGER REFERENCES library_roots(id) ON DELETE CASCADE,
    started_at     INTEGER NOT NULL,
    finished_at    INTEGER,
    files_seen     INTEGER NOT NULL DEFAULT 0,
    files_added    INTEGER NOT NULL DEFAULT 0,
    files_skipped  INTEGER NOT NULL DEFAULT 0,
    files_failed   INTEGER NOT NULL DEFAULT 0,
    status         TEXT    NOT NULL,   -- running | completed | cancelled | failed
    error          TEXT
);

-- Search over filename and tags. Contentless FTS keeps one copy of the text.
--
-- Deviation from overview.md 4.1, which specifies `tokenchars '_-'`. That option does the
-- opposite of what the surrounding prose asks for: it makes `_` and `-` *token* characters
-- rather than separators, so `KICK_808_Distorted-02.wav` indexes as the single token
-- `kick_808_distorted-02` plus `wav`, and searching `808` -- the example the document gives
-- -- matches nothing. Verified against SQLite 3.51.3.
--
-- With the default unicode61 separators both forms work: `808` and `kick` hit as barewords,
-- and the compound form hits as a quoted phrase (`"KICK_808_Distorted-02.wav"`), because
-- FTS5 tokenizes the query the same way it tokenized the document. Migrations are
-- forward-only, so this is fixed here rather than in a V2 nobody would notice was needed.
CREATE VIRTUAL TABLE samples_fts USING fts5(
    filename,
    tags,
    content = '',
    tokenize = "unicode61 remove_diacritics 2"
);
