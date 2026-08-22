#!/usr/bin/env python3
"""Export the LAION-CLAP audio tower to ONNX and record the parity oracle.

**One-time, offline, developer-only.** Nothing in the shipped app runs this, or Python at
all (`overview.md` §1). It exists so that the `.onnx` AudioBank downloads and the reference
embeddings its tests assert against were produced by one process from one checkpoint, and
so that the front-end constants in `src-tauri/src/pipeline/mel.rs` are *checked* against the
model rather than believed about it.

What it produces
----------------

1. ``clap_audio.onnx`` -- the audio tower only. The text tower is dead weight: AudioBank
   never embeds text, and shipping it would roughly double a 200 MB download. Published as
   a release asset; **do not commit it** (``.gitignore`` has ``*.onnx``).
2. ``frontend.json`` -- every mel front-end parameter read off the loaded checkpoint,
   plus the graph's input/output signature. `tests/parity.rs` asserts Rust's constants
   equal these. This is what turns "the front-end probably matches" into a failing test.
3. ``fixtures.json`` + ``reference_embeddings.f32`` -- 20 diverse synthetic signals and the
   embeddings Python CLAP produces for them. This is the parity oracle: `task.md` Phase 3
   requires the Rust front-end feeding this graph to reproduce them at cosine similarity
   **> 0.999**.

Why the fixtures are recipes rather than wavs
---------------------------------------------

Twenty 10-second 48 kHz mono wavs is ~19 MB of binary in git for content fully described by
a line of arithmetic each. So both sides *generate* them from the recipes in ``FIXTURES``
below, and ``fixtures.json`` records a tolerant probe of each waveform -- sixteen samples
and an RMS -- so that a drift between the two generators fails as "the fixture generators
disagree" instead of masquerading as a parity failure. `src-tauri/tests/support/fixtures.rs`
is the Rust half and must be kept in step with this file.

Verified package versions
-------------------------

These are the versions the export was developed against. Pin them; `laion_clap` reaches
into `transformers` and `torchlibrosa` internals and does not tolerate drift, and the whole
value of this script is that its output is reproducible.

    python==3.10.20
    torch==2.4.1
    torchaudio==2.4.1
    torchvision==0.19.1
    torchlibrosa==0.1.0
    librosa==0.10.2.post1
    laion-clap==1.1.6
    transformers==4.30.2
    numpy==1.23.5
    onnx==1.16.2
    onnxruntime==1.19.2

Two of those are traps, and both were found by installing this list from scratch:

* **numpy is 1.23.5, not 1.26.x.** `laion-clap` 1.1.6 hard-pins `numpy==1.23.5`. `pip`
  resolves the conflict by installing one and overwriting it with the other, which is how an
  earlier version of this docstring came to claim 1.26.4 -- a version this environment cannot
  actually hold. A strict resolver refuses outright.
* **torchvision is required and undeclared.** `laion_clap.clap_module.utils` imports
  `torchvision.ops.misc`, but nothing in the dependency tree asks for it, so a clean install
  succeeds and then fails at `import laion_clap`. 0.19.1 is the release paired with torch
  2.4.1.

Python 3.10 is not negotiable either: `transformers` 4.30.2 does not build on 3.12+.

    # `uv` will fetch the interpreter itself; no system Python 3.10 needed.
    uv python install 3.10
    uv venv --python 3.10 .venv-export
    VIRTUAL_ENV=.venv-export uv pip install \\
        torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 laion-clap==1.1.6 \\
        transformers==4.30.2 numpy==1.23.5 librosa==0.10.2.post1 \\
        onnx==1.16.2 onnxruntime==1.19.2

Usage
-----

    python scripts/export_clap_onnx.py --out build/model

Then take the printed SHA-256 and byte count into ``ModelRelease::CURRENT`` in
``src-tauri/src/model/mod.rs``, upload the ``.onnx`` as a release asset, and commit the
fixture files it wrote under ``src-tauri/tests/fixtures/clap/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

# ---------------------------------------------------------------------------------------
# The contract with the Rust side. Every value here is asserted against what the checkpoint
# actually does before anything is exported; a mismatch is a hard failure, because a silent
# one produces embeddings that are wrong and completely plausible.
# ---------------------------------------------------------------------------------------

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 10
MAX_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
WINDOW_SIZE = 1024
HOP_SIZE = 480
MEL_BINS = 64
FMIN = 50.0
FMAX = 14_000.0
MEL_FRAMES = MAX_SAMPLES // HOP_SIZE + 1  # center=True
EMBEDDING_DIM = 512

#: ONNX opset. 17 is the oldest that carries every op the Swin blocks lower to without
#: onnxruntime having to emulate one, and it predates onnxruntime 1.28 comfortably.
OPSET = 17

#: The checkpoint. HTSAT-tiny is the one LAION publishes with the 630k-audioset weights and
#: the one whose audio tower fits the size budget in `overview.md` §7.
AMODEL = "HTSAT-tiny"
TMODEL = "roberta"


# ---------------------------------------------------------------------------------------
# Fixture generation. Mirrored in `src-tauri/tests/support/fixtures.rs`; keep them in step.
# All arithmetic is float64 and cast to float32 exactly once, at the end, so the two
# implementations agree to well within the probe tolerance.
# ---------------------------------------------------------------------------------------

FIXTURES = [
    # Pitched, low to high -- the register where the mel scale's log branch takes over.
    {"name": "sine-55", "kind": "sine", "hz": 55.0, "amp": 0.5, "seconds": 1.0},
    {"name": "sine-220", "kind": "sine", "hz": 220.0, "amp": 0.5, "seconds": 1.0},
    {"name": "sine-1000", "kind": "sine", "hz": 1000.0, "amp": 0.5, "seconds": 2.0},
    {"name": "sine-8000", "kind": "sine", "hz": 8000.0, "amp": 0.3, "seconds": 1.0},
    # Harmonic content: what a real pitched instrument looks like to the filterbank.
    {"name": "saw-110", "kind": "harmonics", "hz": 110.0, "amp": 0.4, "seconds": 2.0, "partials": 24},
    {"name": "saw-440", "kind": "harmonics", "hz": 440.0, "amp": 0.4, "seconds": 1.5, "partials": 12},
    {"name": "chord-major", "kind": "chord", "hz": 261.63, "amp": 0.3, "seconds": 3.0, "semitones": [0, 4, 7]},
    {"name": "chord-minor", "kind": "chord", "hz": 220.0, "amp": 0.3, "seconds": 3.0, "semitones": [0, 3, 7]},
    # Broadband: the other end of the spectral-flatness axis.
    {"name": "noise-white", "kind": "noise", "seed": 1, "amp": 0.35, "seconds": 2.0},
    {"name": "noise-short", "kind": "noise", "seed": 7, "amp": 0.5, "seconds": 0.25},
    {"name": "noise-full", "kind": "noise", "seed": 13, "amp": 0.2, "seconds": 10.0},
    # Percussive one-shots -- the bulk of a real sample library, and the case where
    # `Padding::RepeatPad` does the most work.
    {"name": "kick", "kind": "kick", "hz": 55.0, "amp": 0.9, "seconds": 0.4, "decay": 18.0},
    {"name": "kick-short", "kind": "kick", "hz": 80.0, "amp": 0.9, "seconds": 0.12, "decay": 45.0},
    {"name": "snare", "kind": "snare", "seed": 21, "hz": 190.0, "amp": 0.8, "seconds": 0.3, "decay": 22.0},
    {"name": "hat", "kind": "hat", "seed": 33, "amp": 0.6, "seconds": 0.08, "decay": 90.0},
    {"name": "clicks-120", "kind": "clicks", "bpm": 120.0, "amp": 0.8, "seconds": 4.0, "decay": 300.0},
    # Sweeps and edges.
    {"name": "chirp-up", "kind": "chirp", "hz": 60.0, "hz_end": 12000.0, "amp": 0.4, "seconds": 5.0},
    {"name": "chirp-down", "kind": "chirp", "hz": 12000.0, "hz_end": 60.0, "amp": 0.4, "seconds": 5.0},
    {"name": "silence", "kind": "silence", "seconds": 1.0},
    {"name": "dc-offset", "kind": "dc", "amp": 0.25, "seconds": 1.0},
]


def _xorshift(seed: int, n: int) -> list[float]:
    """The generator in `src-tauri/tests/support/mod.rs`, value for value.

    Hand-rolled in both languages so no PRNG implementation detail sits between them.
    """
    state = seed | 1
    out = []
    mask = (1 << 64) - 1
    for _ in range(n):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        out.append(((state >> 40) / 8_388_608.0) - 1.0)
    return out


def synthesize(spec: dict) -> list[float]:
    """One fixture waveform, as float64. Mirrored in Rust."""
    n = int(spec["seconds"] * SAMPLE_RATE)
    kind = spec["kind"]
    amp = spec.get("amp", 0.0)
    sr = float(SAMPLE_RATE)

    if kind == "silence":
        return [0.0] * n
    if kind == "dc":
        return [amp] * n
    if kind == "sine":
        return [amp * math.sin(2.0 * math.pi * spec["hz"] * i / sr) for i in range(n)]
    if kind == "harmonics":
        # 1/k partials: a band-limited sawtooth, which is what most pitched samples look
        # like to a 64-band filterbank.
        partials = spec["partials"]
        out = []
        for i in range(n):
            acc = 0.0
            for k in range(1, partials + 1):
                hz = spec["hz"] * k
                if hz >= sr / 2.0:
                    break
                acc += math.sin(2.0 * math.pi * hz * i / sr) / k
            out.append(amp * acc)
        return out
    if kind == "chord":
        out = []
        freqs = [spec["hz"] * (2.0 ** (s / 12.0)) for s in spec["semitones"]]
        for i in range(n):
            out.append(amp * sum(math.sin(2.0 * math.pi * f * i / sr) for f in freqs))
        return out
    if kind == "noise":
        return [amp * v for v in _xorshift(spec["seed"], n)]
    if kind == "kick":
        # Pitch-swept sine under an exponential decay: the shape of every kick drum.
        out = []
        phase = 0.0
        for i in range(n):
            t = i / sr
            hz = spec["hz"] * (1.0 + 3.0 * math.exp(-40.0 * t))
            phase += 2.0 * math.pi * hz / sr
            out.append(amp * math.exp(-spec["decay"] * t) * math.sin(phase))
        return out
    if kind == "snare":
        noise = _xorshift(spec["seed"], n)
        out = []
        for i in range(n):
            t = i / sr
            env = math.exp(-spec["decay"] * t)
            body = math.sin(2.0 * math.pi * spec["hz"] * t)
            out.append(amp * env * (0.6 * noise[i] + 0.4 * body))
        return out
    if kind == "hat":
        noise = _xorshift(spec["seed"], n)
        out = []
        prev = 0.0
        for i in range(n):
            t = i / sr
            # One-pole difference: a crude high-pass, enough to put the energy where a
            # closed hi-hat puts it.
            high = noise[i] - prev
            prev = noise[i]
            out.append(amp * math.exp(-spec["decay"] * t) * high)
        return out
    if kind == "clicks":
        period = int(sr * 60.0 / spec["bpm"])
        out = []
        for i in range(n):
            t = (i % period) / sr
            out.append(amp * math.exp(-spec["decay"] * t))
        return out
    if kind == "chirp":
        # Linear frequency sweep; the phase is the integral of the instantaneous frequency.
        f0, f1 = spec["hz"], spec["hz_end"]
        seconds = spec["seconds"]
        out = []
        for i in range(n):
            t = i / sr
            phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / seconds)
            out.append(amp * math.sin(phase))
        return out
    raise SystemExit(f"unknown fixture kind {kind!r}")


def probe(samples: list[float]) -> dict:
    """A drift detector for the two generators, tolerant of a last-ulp `sin` difference.

    Sixteen evenly spaced samples plus an RMS. A hash would be stricter and useless: `sin`
    is allowed to differ in the last bit between libms, and that must not read as a parity
    failure.
    """
    n = len(samples)
    step = max(n // 16, 1)
    return {
        "len": n,
        "at": [float(samples[min(i * step, n - 1)]) for i in range(16)],
        "rms": math.sqrt(sum(s * s for s in samples) / n) if n else 0.0,
    }


def repeat_pad(samples, max_len: int):
    """LAION-CLAP's ``data_filling="repeatpad"``, transcribed.

    Whole copies then zeros -- not a partial final copy. `Padding::RepeatPad` in
    `src-tauri/src/pipeline/mel.rs` is the Rust half.
    """
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(max_len, dtype=np.float32)
    if len(x) >= max_len:
        return x[:max_len]
    repeats = max(max_len // len(x), 1)
    x = np.tile(x, repeats)
    return np.pad(x, (0, max_len - len(x)), mode="constant")[:max_len]


# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------


class AudioTower:
    """Wraps HTSAT so the graph starts at the log-mel spectrogram, not at the waveform.

    Two reasons the mel front-end is outside the graph:

    * `overview.md` §3.3 shares one STFT between the DSP descriptors and the mel bands. If
      the graph did its own, AudioBank would pay for the transform twice per file.
    * `torchlibrosa`'s STFT exports as a conv1d over a 1024x513 constant, which is ~2 MB of
      weights and slower than `realfft` on CPU.

    The cost is that the front-end becomes AudioBank's responsibility, which is exactly what
    the parity gate exists to police.

    **This reaches into `laion_clap` internals.** They move between versions; if the export
    fails here, read `laion_clap/clap_module/htsat.py` at the pinned version above and
    follow ``HTSAT_Swin_Transformer.forward`` from ``self.logmel_extractor`` onward.
    """

    def __init__(self, clap):
        import torch

        self.torch = torch
        self.clap = clap

    def build(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        model = self.clap.model
        branch = model.audio_branch
        projection = model.audio_projection

        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = branch
                self.projection = projection

            def forward(self, logmel):
                # `logmel` is (B, 1, frames, mels) -- torchlibrosa's own layout, which is
                # what `LogmelFilterBank` hands `HTSAT_Swin_Transformer.forward`.
                x = logmel.transpose(1, 3)
                x = self.branch.bn0(x)
                x = x.transpose(1, 3)
                # `spec_augmenter` and `mixup` are training-only and are skipped here by
                # construction rather than by an `if self.training`, so the exported graph
                # cannot contain them.
                x = self.branch.reshape_wav2img(x)
                embedding = self.branch.forward_features(x)["embedding"]
                return F.normalize(self.projection(embedding), dim=-1)

        tower = Tower().eval()
        for p in tower.parameters():
            p.requires_grad_(False)
        return tower

    def logmel(self, waveform):
        """The reference front-end: CLAP's own extractors, not a reimplementation."""
        branch = self.clap.model.audio_branch
        x = self.torch.as_tensor(waveform, dtype=self.torch.float32).unsqueeze(0)
        with self.torch.no_grad():
            return branch.logmel_extractor(branch.spectrogram_extractor(x))


def read_frontend(clap) -> dict:
    """Reads the front-end parameters off the loaded checkpoint.

    Read, not assumed. This dict is what `tests/parity.rs` compares Rust's
    ``FrontEnd::SPEC`` against, so anything here that is a guess defeats the purpose.
    """
    branch = clap.model.audio_branch
    spec = branch.spectrogram_extractor
    mel = branch.logmel_extractor
    stft = spec.stft

    def attr(obj, *names, default=None):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    return {
        "sample_rate": SAMPLE_RATE,
        "window_size": int(attr(stft, "n_fft", default=WINDOW_SIZE)),
        "hop_size": int(attr(stft, "hop_length", default=HOP_SIZE)),
        "mel_bins": int(mel.melW.shape[1]),
        "fmin": float(attr(mel, "fmin", default=FMIN)),
        "fmax": float(attr(mel, "fmax", default=FMAX)),
        "amin": float(attr(mel, "amin", default=1e-10)),
        "ref_value": float(attr(mel, "ref", default=1.0)),
        "top_db": (lambda v: None if v is None else float(v))(attr(mel, "top_db")),
        "center": bool(attr(stft, "center", default=True)),
        "pad_mode": str(attr(stft, "pad_mode", default="reflect")),
        "power": float(attr(spec, "power", default=2.0)),
        # `librosa.filters.mel` defaults, which is what torchlibrosa calls. Recorded rather
        # than detected: the filterbank is a materialized matrix by this point and the
        # scale that produced it is no longer an attribute of anything.
        "htk": False,
        "slaney_norm": True,
        "frames": MEL_FRAMES,
        "padding": "repeatpad",
        "window_seconds": WINDOW_SECONDS,
    }


def check_contract(frontend: dict) -> None:
    """Fails the export if the checkpoint is not the front-end Rust implements.

    Better here, loudly, than as a cosine similarity of 0.7 in CI with no clue why.
    """
    expected = {
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "mel_bins": MEL_BINS,
        "fmin": FMIN,
        "fmax": FMAX,
        "center": True,
        "pad_mode": "reflect",
        "power": 2.0,
        "top_db": None,
        "ref_value": 1.0,
        "amin": 1e-10,
    }
    problems = [
        f"  {key}: checkpoint says {frontend[key]!r}, src/pipeline/mel.rs assumes {want!r}"
        for key, want in expected.items()
        if frontend[key] != want
    ]
    if problems:
        raise SystemExit(
            "the checkpoint's front-end does not match this build's constants:\n"
            + "\n".join(problems)
            + "\n\nFix src-tauri/src/pipeline/mel.rs to match, then re-run. Do not "
            "'fix' this check."
        )


def sha256_file(path: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def write_recipes(out: pathlib.Path) -> int:
    """Writes ``fixtures.json`` with recipes and waveform probes, and nothing else.

    Split out so it needs neither torch nor the checkpoint. The embeddings are the half of
    the oracle that requires the model; the recipes are the half that lets
    `tests/parity.rs` prove the two generators still agree, and there is no reason for the
    second to wait on the first.
    """
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in FIXTURES:
        entry = dict(spec)
        entry["probe"] = probe(synthesize(spec))
        manifest.append(entry)
        print(f"  {spec['name']:<14} n={entry['probe']['len']:>7} rms={entry['probe']['rms']:.6f}")

    (out / "fixtures.json").write_text(
        json.dumps(
            {
                "note": "Generated by scripts/export_clap_onnx.py. Do not hand-edit.",
                "sample_rate": SAMPLE_RATE,
                "embedding_dim": EMBEDDING_DIM,
                "fixtures": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}/fixtures.json ({len(manifest)} fixtures, no embeddings)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build/model"),
                        help="where to write clap_audio.onnx")
    parser.add_argument("--fixtures", type=pathlib.Path,
                        default=pathlib.Path("src-tauri/tests/fixtures/clap"),
                        help="where to write the committed parity fixtures")
    parser.add_argument("--ckpt", type=pathlib.Path, default=None,
                        help="a local .pt checkpoint; downloads LAION's if omitted")
    parser.add_argument("--skip-export", action="store_true",
                        help="regenerate fixtures only, against an existing checkpoint")
    parser.add_argument("--recipes-only", action="store_true",
                        help="write fixtures.json (recipes + waveform probes) and stop. "
                             "Needs no checkpoint and no torch; this is how the Rust-side "
                             "generator is kept honest without a 2 GB download.")
    args = parser.parse_args()

    if args.recipes_only:
        return write_recipes(args.fixtures)

    import numpy as np
    import torch
    import laion_clap

    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    args.fixtures.mkdir(parents=True, exist_ok=True)

    print(f"loading {AMODEL} ...", flush=True)
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=AMODEL, tmodel=TMODEL)
    clap.load_ckpt(str(args.ckpt) if args.ckpt else None)
    clap.eval()

    frontend = read_frontend(clap)
    check_contract(frontend)
    print(json.dumps(frontend, indent=2))

    wrapper = AudioTower(clap)
    tower = wrapper.build()

    onnx_path = args.out / "clap_audio.onnx"
    if not args.skip_export:
        dummy = torch.zeros(1, 1, MEL_FRAMES, MEL_BINS, dtype=torch.float32)
        with torch.no_grad():
            out = tower(dummy)
        if out.shape[-1] != EMBEDDING_DIM:
            raise SystemExit(
                f"the audio tower emits {out.shape[-1]} dimensions; "
                f"src-tauri/src/lib.rs pins EMBEDDING_DIM at {EMBEDDING_DIM}"
            )

        print(f"exporting to {onnx_path} (opset {OPSET}) ...", flush=True)
        torch.onnx.export(
            tower,
            dummy,
            str(onnx_path),
            input_names=["logmel"],
            output_names=["embedding"],
            # Batch is dynamic so the Phase 4 batcher can stack 16-32 spectrograms into one
            # run(); the mel and frame axes are **fixed** on purpose, so that a graph which
            # would silently accept the wrong front-end cannot load at all
            # (`model::session::layout_of`).
            dynamic_axes={"logmel": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )

        digest, size = sha256_file(onnx_path)
        print()
        print("Paste into ModelRelease::CURRENT in src-tauri/src/model/mod.rs:")
        print(f'        sha256: "{digest}",')
        print(f"        bytes: Some({size}),")
        print()

    # ---- the parity oracle -------------------------------------------------------------
    print(f"generating {len(FIXTURES)} reference embeddings ...", flush=True)
    manifest = []
    embeddings = np.zeros((len(FIXTURES), EMBEDDING_DIM), dtype=np.float32)

    for index, spec in enumerate(FIXTURES):
        samples = synthesize(spec)
        padded = repeat_pad(samples, MAX_SAMPLES)
        mel = wrapper.logmel(padded)
        if tuple(mel.shape) != (1, 1, MEL_FRAMES, MEL_BINS):
            raise SystemExit(
                f"{spec['name']}: reference mel is {tuple(mel.shape)}, "
                f"expected (1, 1, {MEL_FRAMES}, {MEL_BINS})"
            )
        with torch.no_grad():
            embeddings[index] = tower(mel).cpu().numpy()[0]

        entry = dict(spec)
        entry["probe"] = probe(samples)
        manifest.append(entry)
        print(f"  {spec['name']:<14} |x|={float(np.linalg.norm(embeddings[index])):.6f}")

    (args.fixtures / "reference_embeddings.f32").write_bytes(embeddings.tobytes())
    (args.fixtures / "fixtures.json").write_text(
        json.dumps(
            {
                "note": "Generated by scripts/export_clap_onnx.py. Do not hand-edit.",
                "sample_rate": SAMPLE_RATE,
                "embedding_dim": EMBEDDING_DIM,
                "fixtures": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    (args.fixtures / "frontend.json").write_text(json.dumps(frontend, indent=2) + "\n")

    print()
    print(f"wrote {args.fixtures}/fixtures.json, frontend.json, reference_embeddings.f32")
    print("commit those three; publish the .onnx as a release asset (never commit it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
