#!/usr/bin/env python3
"""Build the tiny ONNX graphs that `src-tauri/tests/session.rs` runs against.

**Developer-only, and run once.** The outputs are committed; this exists so they can be
regenerated and so that what is in them is readable rather than a mystery blob.

Why these exist
---------------

The real CLAP model is a 200 MB release asset that CI will never have, so every test of
`model::session` would otherwise be a test of code paths that have never met ONNX Runtime.
These graphs are ~130 KB and structurally faithful where it matters: the input signature,
the output width, and the dynamic batch axis are exactly the real export's, so session
construction, execution-provider selection, the warmup, layout detection, batching, and
extraction are all exercised for real. What they are *not* is CLAP -- they compute
nonsense, deterministically. Parity is `tests/parity.rs`'s job and needs the real model.

The graph is `ReduceMean` over the frame axis, then `MatMul` with a fixed pseudo-random
`[64, 512]`, then `Add` of a bias. Cheap, but not degenerate: every one of the 64 mel bands
reaches every one of the 512 outputs, so a front-end that transposed its spectrogram or
filled the wrong axis produces visibly different numbers rather than the same ones.

    python3 scripts/make_session_fixtures.py

Needs `onnx` (`pip install onnx==1.19.1`) and nothing else -- deliberately not `torch`.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Must match src-tauri/src/pipeline/mel.rs and src-tauri/src/lib.rs.
MEL_FRAMES = 480_000 // 480 + 1
MEL_BINS = 64
EMBEDDING_DIM = 512
OPSET = 17

OUT = pathlib.Path("src-tauri/tests/fixtures/session")


def weights() -> tuple[np.ndarray, np.ndarray]:
    """Fixed pseudo-random projection. Seeded so the committed graphs are reproducible."""
    rng = np.random.default_rng(20240321)
    w = rng.standard_normal((MEL_BINS, EMBEDDING_DIM)).astype(np.float32) / 8.0
    b = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) / 8.0
    return w, b


def build(path: pathlib.Path, *, frames_major: bool, dynamic_mels: bool = False) -> None:
    """Writes one graph.

    `frames_major` picks between HTSAT's own `[B, 1, T, 64]` and `overview.md` §3.4's
    `[B, 1, 64, T]`; `model::session::layout_of` has to tell them apart and transpose for
    the second. `dynamic_mels` builds the graph that must be *rejected* -- one that would
    happily accept a spectrogram with the wrong number of bands.
    """
    w, b = weights()
    frame_axis, mel_axis = (2, 3) if frames_major else (3, 2)
    shape = [None, 1, None, None]
    shape[frame_axis] = MEL_FRAMES
    shape[mel_axis] = "mels" if dynamic_mels else MEL_BINS
    shape[0] = "batch"

    nodes = [
        # Average away the time axis: [B, 1, T, 64] -> [B, 1, 1, 64] (or the transpose).
        # `axes` is an attribute, not an input: it only became an input in opset 18, and
        # this graph is pinned to 17 to match the real export.
        helper.make_node("ReduceMean", ["logmel"], ["pooled"], axes=[frame_axis], keepdims=0),
        helper.make_node("Reshape", ["pooled", "flat_shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "W"], ["projected"]),
        helper.make_node("Add", ["projected", "B"], ["embedding"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array([-1, MEL_BINS], dtype=np.int64), "flat_shape"),
        numpy_helper.from_array(w, "W"),
        numpy_helper.from_array(b, "B"),
    ]

    graph = helper.make_graph(
        nodes,
        "audiobank_session_fixture",
        [helper.make_tensor_value_info("logmel", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(
                "embedding", TensorProto.FLOAT, ["batch", EMBEDDING_DIM]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)]
    )
    model.ir_version = 9  # onnxruntime 1.28 rejects ir_version 11 from onnx 1.19.
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(f"  {path}  ({path.stat().st_size:,} bytes)  input {shape}")


def main() -> int:
    build(OUT / "frames_major.onnx", frames_major=True)
    build(OUT / "mels_major.onnx", frames_major=False)
    build(OUT / "dynamic_mels.onnx", frames_major=True, dynamic_mels=True)

    # The expected embedding for an all-zeros spectrogram, so `tests/session.rs` can prove
    # it extracted the right numbers rather than merely a tensor of the right size.
    w, b = weights()
    zeros = np.zeros(MEL_BINS, dtype=np.float32) @ w + b
    zeros = zeros / np.linalg.norm(zeros)
    (OUT / "silence_embedding.f32").write_bytes(zeros.astype(np.float32).tobytes())
    print(f"  {OUT}/silence_embedding.f32  (L2-normalized reference for a zero input)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
