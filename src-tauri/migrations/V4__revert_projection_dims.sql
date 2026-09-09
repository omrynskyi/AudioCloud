-- V4: revert V3 -- back to exactly one active projection, forever.
--
-- V3's two-independently-active-layouts design was replaced by a client-side flatten of the
-- active 3D layout for 2D mode instead: an independent 2D fit is a fresh, unconstrained
-- optimization with far less room than 3D to keep true neighborhoods apart, and testing
-- found exactly that -- samples with no real similarity landing right next to each other on
-- the 2D map. Nothing needs a `dims`-scoped active slot any more.
--
-- Migrations are forward-only (overview.md 4.5), so V3 is not deleted -- a database that
-- already applied it (including this project's own during development) must still find it
-- on disk with the same content, or refinery refuses to start. This migration undoes its
-- effect instead, which is the correct way to retire a migration that shipped.
--
-- Whichever run was the active *3D* one keeps the single "active" slot; a database that
-- happened to have only a 2D run active (unlikely, since 3D existed first, but not
-- impossible) falls back to whatever was active, rather than being left with no map at all.
UPDATE projection_runs SET is_active = 0
WHERE id <> (
    SELECT id FROM projection_runs
    WHERE is_active = 1
    ORDER BY (dims <> 3), id DESC
    LIMIT 1
);

-- Everything but the kept run is abandoned data now -- a superseded run, or the independent
-- 2D fit this migration exists to retire. Cascades to `projections` via the foreign key.
DELETE FROM projection_runs WHERE is_active = 0;

DROP INDEX idx_projection_active;

CREATE UNIQUE INDEX idx_projection_active
    ON projection_runs(is_active) WHERE is_active = 1;

ALTER TABLE projection_runs DROP COLUMN dims;
