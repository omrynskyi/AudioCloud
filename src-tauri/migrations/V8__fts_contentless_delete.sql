-- Make FTS row replacement and cascade removal reliable.
--
-- V7's first shipped form added `path` to the index but retained a contentless FTS table that
-- cannot delete by row id. V7 was subsequently edited before its checksum was frozen, so V8 is
-- deliberately a complete forward rebuild rather than an in-place alteration: FTS5 cannot add
-- `contentless_delete` to an existing virtual table. The source table remains authoritative;
-- this only recreates its derived search index.

DROP TRIGGER IF EXISTS samples_fts_delete;
DROP TABLE samples_fts;

CREATE VIRTUAL TABLE samples_fts USING fts5(
    filename,
    path,
    tags,
    content = '',
    contentless_delete = 1,
    tokenize = "unicode61 remove_diacritics 2"
);

INSERT INTO samples_fts (rowid, filename, path, tags)
SELECT
    s.id,
    s.filename,
    s.rel_path,
    COALESCE(
        (
            SELECT group_concat(t.name, ' ' ORDER BY t.name COLLATE NOCASE)
            FROM tags t
            JOIN sample_tags st ON st.tag_id = t.id
            WHERE st.sample_id = s.id
        ),
        ''
    )
FROM samples s;

-- Foreign-key cascades also fire this trigger, so removing a library root cannot leave stale
-- rows behind in the contentless index.
CREATE TRIGGER samples_fts_delete AFTER DELETE ON samples BEGIN
    DELETE FROM samples_fts WHERE rowid = OLD.id;
END;
