"""Feasibility spike: probe ASLP-lab/SongFormer as an autohotcue structure backend.

Run a compatibility scan (no model load):

    uv run --with transformers --with muq python scripts/songformer_spike.py --check

Run inference on one or more tracks:

    uv run --with transformers --with muq python scripts/songformer_spike.py --device mps "/path/track.opus"

    uv run --with transformers --with muq python scripts/songformer_spike.py --device mps "/path/track.opus" [more tracks]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

MODEL_ID = "ASLP-lab/SongFormer"
REVISION = os.environ.get(
    "AUTOHOTCUE_SONGFORMER_REVISION",
    "a75880ed1b7375ac71860ec6c4fc9c899cf99515",
)
MODEL_SR = 24000

# Landmine patterns for --check scan.
_TORCH_LOAD = re.compile(r"torch\.load\s*\(")
_WEIGHTS_ONLY = re.compile(r"weights_only\s*=")
_NUMPY2_PATTERNS = [
    ("np.NaN", re.compile(r"\bnp\.NaN\b")),
    ("np.float_", re.compile(r"\bnp\.float_\b")),
    ("np.int_", re.compile(r"\bnp\.int_\b")),
    ("np.alltrue", re.compile(r"\bnp\.alltrue\b")),
    ("np.sometrue", re.compile(r"\bnp\.sometrue\b")),
]
_ARRAY_COPY_FALSE = re.compile(r"np\.array\s*\([^)]*copy\s*=\s*False")
_TORCHAUDIO_IMPORT = re.compile(r"^\s*(?:import\s+torchaudio|from\s+torchaudio\b)")
_MUSICFM_IMPORT = re.compile(r"^\s*(?:import\s+musicfm|from\s+musicfm\b)")


def _fix_ssl_env() -> None:
    """A socket-firewall wrapper overrides SSL_CERT_FILE with a CA that cannot
    validate huggingface.co; re-point trust to certifi from inside the process."""
    import certifi

    cert = os.environ.get("SSL_CERT_FILE", "")
    if "socketFirewallCa" in cert or not Path(cert or "/nonexistent").exists():
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.pop("SSL_CERT_DIR", None)


def _snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(MODEL_ID, revision=REVISION),
    )


def cmd_check() -> None:
    snapshot = _snapshot_dir()
    total_bytes = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())
    print(f"snapshot: {snapshot}")
    print(f"total_size_bytes: {total_bytes}")

    counts = {
        "torch_load": 0,
        "torch_load_no_weights_only": 0,
        "numpy2": 0,
        "array_copy_false": 0,
        "torchaudio_import": 0,
        "musicfm_import": 0,
    }

    for py_path in sorted(snapshot.rglob("*.py")):
        rel = py_path.relative_to(snapshot)
        lines = py_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            loc = f"{rel}:{lineno}"

            if _TORCH_LOAD.search(line):
                counts["torch_load"] += 1
                has_wo = bool(_WEIGHTS_ONLY.search(line))
                if not has_wo:
                    counts["torch_load_no_weights_only"] += 1
                print(f"{loc}: torch.load( weights_only={'yes' if has_wo else 'NO'})")

            for name, pat in _NUMPY2_PATTERNS:
                if pat.search(line):
                    counts["numpy2"] += 1
                    print(f"{loc}: numpy2 removal: {name}")

            if _ARRAY_COPY_FALSE.search(line):
                counts["array_copy_false"] += 1
                print(f"{loc}: np.array(..., copy=False)")

            if _TORCHAUDIO_IMPORT.search(line):
                counts["torchaudio_import"] += 1
                print(f"{loc}: torchaudio import")

            if _MUSICFM_IMPORT.search(line):
                counts["musicfm_import"] += 1
                print(f"{loc}: musicfm import")

    print(
        "SUMMARY: "
        f"torch_load={counts['torch_load']} "
        f"(no_weights_only={counts['torch_load_no_weights_only']}), "
        f"numpy2={counts['numpy2']}, "
        f"array_copy_false={counts['array_copy_false']}, "
        f"torchaudio_import={counts['torchaudio_import']}, "
        f"musicfm_import={counts['musicfm_import']}",
    )


def _stub_msaf() -> None:
    """model.py imports msaf only for its eval-time cal_metrics(); stub it so
    inference does not require the (numpy-2-incompatible) msaf package."""
    import importlib.machinery
    import types

    if "msaf" in sys.modules:
        return
    msaf = types.ModuleType("msaf")
    msaf_eval = types.ModuleType("msaf.eval")

    def _unavailable(**kwargs):
        raise RuntimeError("msaf is stubbed out; cal_metrics() is unavailable")

    msaf_eval.compute_results = _unavailable
    msaf.eval = msaf_eval
    msaf.__spec__ = importlib.machinery.ModuleSpec("msaf", None)
    msaf_eval.__spec__ = importlib.machinery.ModuleSpec("msaf.eval", None)
    sys.modules["msaf"] = msaf
    sys.modules["msaf.eval"] = msaf_eval


def _resolve_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return choice


def _load_model(device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    # modeling_songformer.__init__ reads SONGFORMER_LOCAL_DIR to find its
    # bundled muq_config2.json / msd_stats.json.
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    os.environ.setdefault("SONGFORMER_LOCAL_DIR", snapshot)

    load_strategy = "direct"
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=REVISION,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_files_only=True,
        )
        sys.path.insert(0, snapshot)
        load_strategy = f"sys.path.insert(0, {snapshot!r})"
        print(f"direct load failed ({exc!r}); retrying with {load_strategy}")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=REVISION,
        )

    print(f"load_strategy: {load_strategy}")
    model.eval()
    model.to(device)
    return model


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _run_track(model, device: str, track_path: str) -> None:
    import librosa
    import numpy as np
    import torch

    from autohotcue.analysis import SR, decode

    y = decode(track_path, sr=SR)
    duration_s = len(y) / SR
    y_model = librosa.resample(y, orig_sr=SR, target_sr=MODEL_SR)
    waveform = np.asarray(y_model, dtype=np.float32)

    t0 = time.perf_counter()
    with torch.inference_mode():
        segments = model(waveform)
    forward_s = time.perf_counter() - t0

    rss = _peak_rss_bytes()
    print(
        f"track={track_path!r} duration_s={duration_s:.3f} device={device} "
        f"forward_s={forward_s:.3f} peak_rss_bytes={rss} "
        f"segments={json.dumps(segments)}",
    )


def _empty_mps_cache() -> None:
    import torch

    try:
        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def cmd_run(device_choice: str, tracks: list[str]) -> None:
    device = _resolve_device(device_choice)
    _stub_msaf()
    model = _load_model(device)

    for track_path in tracks:
        try:
            _run_track(model, device, track_path)
        except Exception as exc:
            print(f"ERROR track={track_path!r}: {exc!r}", file=sys.stderr)
            if device == "mps":
                print(
                    "MPS failure — retry with: --device cpu",
                    file=sys.stderr,
                )
        _empty_mps_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SongFormer integration feasibility spike",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Download snapshot and scan for compatibility landmines (no model load)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "tracks",
        nargs="*",
        help="Audio file paths to analyze",
    )
    args = parser.parse_args()
    _fix_ssl_env()

    if args.check:
        cmd_check()
        return

    if not args.tracks:
        parser.error("provide at least one track path, or use --check")

    cmd_run(args.device, args.tracks)


if __name__ == "__main__":
    main()
