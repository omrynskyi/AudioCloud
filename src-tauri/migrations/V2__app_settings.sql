-- V2: a small key/value store for settings that must survive a restart
-- (task.md Phase 9 -- audio device, gain). Everything else the Settings panel shows is
-- either derived (model status) or already a durable table (projection params live on the
-- `projection_runs` row a re-fit creates).
CREATE TABLE app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
