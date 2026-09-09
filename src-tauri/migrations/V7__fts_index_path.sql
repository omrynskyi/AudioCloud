-- Rebuild the full-text index: add `rel_path`, make deletes real, and repair what the missing
-- deletes had already corrupted.
--
-- Three problems, one table, so one migration.
--
-- **1. The folders were not indexed.** V1 indexed `filename` and `tags` only, which for a
-- sample library throws away most of the organizational signal there is: the folders are the
-- taxonomy. On the development library `kicks` matched nothing while `Kicks/` held the kicks,
-- and no query could reach `[FREE VERSION] @PRODBY.XERO UK UNDERGROUND DRUM KIT` at all.
-- Anyone who has filed samples into folders has already told us what they are.
--
-- **2. Nothing ever deleted an fts row.** `samples_fts` is a virtual table, so the
-- `ON DELETE CASCADE` that clears `samples` when a library root is removed does not reach it,
-- and no code deleted from it either. Removing a root therefore left every one of its rows in
-- the index, and because `samples.id` is a plain `INTEGER PRIMARY KEY`, the *next* scan
-- reissued those same ids to different files. The index then answered for a library that no
-- longer existed: on the development database, searching `kick` returned
-- `@prodby.xero Snare - Razor.wav` and thirty-four others like it -- not stale results, results
-- for files whose names never contained the word. `contentless_delete = 1` (fts5 in SQLite
-- 3.43+) makes `DELETE FROM samples_fts WHERE rowid = ?` work on a contentless table, and the
-- trigger below fires on cascade deletes as well as direct ones.
--
-- **3. Deleting by re-supplying values was a loaded gun.** Without `contentless_delete`, a row
-- is removed by handing fts5 the exact text it was indexed with; get it wrong and the index
-- corrupts silently rather than failing. `db::writer::reindex_sample` did that dance on every
-- retag, and its own comment says what happens when the values disagree. With real deletes it
-- does not have to know what it is replacing.
--
-- The rebuild from `samples` is what repairs (2) for existing databases -- fts5 has no
-- `ADD COLUMN` and a contentless table keeps no copy of the text, so the rows had to be
-- rebuilt from scratch for (1) regardless, and doing so discards whatever the old index held.
--
-- `path` carries the whole `rel_path`, filename included, rather than just the directory part.
-- The duplicated tokens cost nothing a query can observe -- `MATCH` is a disjunction over
-- columns, so a bareword that hit `filename` already hit -- and the alternative is string
-- surgery here and in the writer, kept in agreement forever, to save a little index. `filename`
-- stays its own column so a column-scoped `filename:kick` can still mean the file, not its
-- folder.
--
-- The `tags` rebuild reproduces `db::writer::fts_tag_text`: names joined by a space, ordered by
-- name, case-insensitively.

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

-- A trigger rather than a delete alongside each `DELETE FROM samples`, because the deletes
-- that caused the corruption were not written as deletes at all: they were a cascade from
-- `library_roots`. SQLite fires triggers for rows removed by a foreign-key action, so this
-- covers the cascade, any future direct delete, and anything else that ever removes a sample.
CREATE TRIGGER samples_fts_delete AFTER DELETE ON samples BEGIN
    DELETE FROM samples_fts WHERE rowid = OLD.id;
END;
