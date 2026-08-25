-- V3: two independently-active layouts, one per dimensionality.
--
-- Every projection_runs row up to now is a 3D fit, so backfilling `dims = 3` describes
-- exactly what those rows already are. The partial unique index moves from "one active row,
-- period" to "one active row per dims" -- a 2D map and a 3D map can each be active at once,
-- built and re-fit independently, with the mode switcher in the UI deciding which is read.
ALTER TABLE projection_runs ADD COLUMN dims INTEGER NOT NULL DEFAULT 3;

DROP INDEX idx_projection_active;

CREATE UNIQUE INDEX idx_projection_active
    ON projection_runs(dims) WHERE is_active = 1;
