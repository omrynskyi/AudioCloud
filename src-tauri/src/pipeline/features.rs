//! DSP feature extraction: `realfft` framing plus the descriptor set.
//!
//! Peak, RMS, LUFS, spectral centroid, spectral flatness, ZCR, onset density, BPM and key
//! all come out of a single pass over the same frames -- the mel front-end that feeds
//! inference is computed here too, and Phase 3's parity gate exists because a front-end
//! that quietly disagrees with the ONNX graph produces embeddings that are wrong in a way
//! nothing downstream can detect.
//!
//! Populated in Phase 2.
