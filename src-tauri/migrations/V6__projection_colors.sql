-- V6: a per-point color, alongside its position.
--
-- Nullable, and on `projections` itself rather than a side table: color is 1:1 with a
-- position row and always read together with it (`queries::active_projection_colors` walks
-- the exact same join as `active_projection_points`), so a second table would only buy a
-- join nobody wants. NULL is the honest state for every row here today, and stays honest
-- for any row a non-color-fitting algorithm (PCA, UMAP) or an incremental placement writes
-- from here on -- the wire format's NaN-per-cell convention (see `ipc::binary`) is what
-- turns that NULL into "use the default color" on the frontend, the same way a sample with
-- no DSP features already does for `get_feature_column`.
ALTER TABLE projections ADD COLUMN r REAL;
ALTER TABLE projections ADD COLUMN g REAL;
ALTER TABLE projections ADD COLUMN b REAL;
