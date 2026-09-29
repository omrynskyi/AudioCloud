-- V9: the active embedder changed from the 1,024-value log-mel fingerprint to a 45-value
-- MFCC-and-envelope description of the whole sound (`crate::pipeline::mfcc_embed`), at a
-- different vector width (`crate::EMBEDDING_DIM`). Same situation as V5: every row's
-- `emb_offset`/`emb_len` describes a byte range in `embeddings.bin` written at the *old*
-- width, and read at the new one it is a wrong number of raw bytes reinterpreted as a vector.
--
-- So every previously-embedded row goes back to `decoded` with its offset cleared, and the
-- next scan re-embeds it under the new embedder. The projection run that is currently
-- active keeps its coordinates, so the map still draws until the re-fit that follows the
-- re-embed replaces it. `embeddings.bin` itself is untouched: the old bytes become dead
-- space, which `EmbeddingStore::compact` reclaims on request.
UPDATE samples
SET status = 'decoded', emb_offset = NULL, emb_len = NULL
WHERE status = 'embedded';
