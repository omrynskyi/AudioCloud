-- V5: the active embedder changed from a CLAP audio-tower model to a downsampled log-mel
-- fingerprint, at a different vector width (`crate::EMBEDDING_DIM`). Every row's
-- `emb_offset`/`emb_len` describes a byte range in `embeddings.bin` written at the *old*
-- width; read at the new one, that range is not a differently-shaped vector, it is a plain
-- wrong number of raw bytes reinterpreted as one.
--
-- So every previously-embedded row is sent back to `decoded` with its offset cleared -- the
-- same state `upsert_samples` already puts a changed file into (`writer.rs`), and the same
-- one a scan with no embedder available leaves a row in. The next scan re-embeds it, this
-- time under the new embedder. Nothing here touches `embeddings.bin` itself: the bytes an
-- old offset pointed at become dead space rather than being reclaimed, which is what
-- `EmbeddingStore::compact` exists to do on request, not something a migration should do
-- unasked to a file this large.
UPDATE samples
SET status = 'decoded', emb_offset = NULL, emb_len = NULL
WHERE status = 'embedded';
