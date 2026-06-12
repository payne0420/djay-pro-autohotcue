"""Optional SongFormer structure backend (CPU by default; HF remote code pinned by revision)."""
from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

import numpy as np

from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis, energy_ranks_for_segments

_MODEL_ID = "ASLP-lab/SongFormer"
_HF_REVISION = "a75880ed1b7375ac71860ec6c4fc9c899cf99515"
_MODEL_SR = 24000
SONGFORMER_LABELS = frozenset({"intro", "verse", "chorus", "bridge", "inst", "outro"})
_LABEL_MAP = {"pre-chorus": "verse", "prechorus": "verse"}
_DROP_LABELS = frozenset({"silence"})

_model = None
_model_device: str | None = None
_model_revision: str | None = None


def _import_error(exc: BaseException | None = None) -> RuntimeError:
    msg = (
        "ml-songformer requires optional deps installed:\n"
        "  uv sync --extra songformer\n"
        "First run downloads the SongFormer model (~2.8 GB one-time)."
    )
    if exc is not None:
        return RuntimeError(f"{msg}\n({type(exc).__name__}: {exc})")
    return RuntimeError(msg)


def _fix_ssl_env() -> None:
    """A socket-firewall wrapper overrides SSL_CERT_FILE with a CA that cannot
    validate huggingface.co; re-point trust to certifi from inside the process."""
    import certifi

    cert = os.environ.get("SSL_CERT_FILE", "")
    if "socketFirewallCa" in cert or not Path(cert or "/nonexistent").exists():
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.pop("SSL_CERT_DIR", None)


def _stub_msaf() -> None:
    """model.py imports msaf only for its eval-time cal_metrics(); stub it so
    inference does not require the (numpy-2-incompatible) msaf package."""
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


def _resolve_device() -> str:
    # Env var intentionally takes precedence over analyze()'s device flag (mirrors ml-allin1).
    return os.environ.get("AUTOHOTCUE_SONGFORMER_DEVICE", "cpu")


def _hf_revision() -> str:
    return os.environ.get("AUTOHOTCUE_SONGFORMER_REVISION", _HF_REVISION)


def _snapshot_download(revision: str) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(_MODEL_ID, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        _fix_ssl_env()
        return snapshot_download(_MODEL_ID, revision=revision)


def _get_model(device: str):
    global _model, _model_device, _model_revision
    revision = _hf_revision()
    if _model is not None and (_model_device != device or _model_revision != revision):
        # Remote-code modules and SONGFORMER_LOCAL_DIR are process-global; a second
        # load would mix snapshots and double peak memory.
        raise RuntimeError(
            f"ml-songformer model already loaded for device={_model_device!r} "
            f"revision={_model_revision!r}; one device/revision per process — "
            "restart to switch"
        )
    if _model is None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise _import_error(exc) from exc

        snapshot = _snapshot_download(revision)
        os.environ["SONGFORMER_LOCAL_DIR"] = snapshot
        _stub_msaf()

        try:
            model = AutoModel.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                revision=revision,
            )
        except (ImportError, ModuleNotFoundError):
            sys.path.insert(0, snapshot)
            model = AutoModel.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                revision=revision,
            )

        model.eval()
        model.to(device)
        _model = model
        _model_device = device
        _model_revision = revision
    return _model


def _map_songformer_segments(raw_segments: list, duration_s: float) -> list[Segment]:
    _known_labels = SONGFORMER_LABELS | _DROP_LABELS
    segments: list[Segment] = []
    for seg in raw_segments:
        label = str(seg["label"]).lower()
        label = _LABEL_MAP.get(label, label)
        if label not in _known_labels:
            raise ValueError(
                f"unknown songformer segment label {seg['label']!r} "
                f"(expected one of {sorted(SONGFORMER_LABELS)})"
            )
        start = float(seg["start"])
        end = float(seg["end"])
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"non-finite segment boundary for {seg['label']!r}: "
                f"start={seg['start']!r}, end={seg['end']!r}"
            )
        if label in _DROP_LABELS:
            continue
        start = max(0.0, start)
        end = min(duration_s, end)
        if end > start:
            segments.append(Segment(start=start, end=end, label=label))
    return segments


def segment_structure_songformer(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
) -> StructureAnalysis:
    """Structure segments from HuggingFace ASLP-lab/SongFormer."""
    del path  # ffmpeg-decoded array is the input; path kept for API stability
    import librosa

    try:
        import torch
    except ImportError as exc:
        raise _import_error(exc) from exc

    device = _resolve_device()
    model = _get_model(device)
    y_model = librosa.resample(y, orig_sr=sr, target_sr=_MODEL_SR)
    waveform = np.asarray(y_model, dtype=np.float32)
    try:
        with torch.inference_mode():
            raw_segments = model(waveform)
        segments = _map_songformer_segments(raw_segments, beat.duration_s)
        segments = energy_ranks_for_segments(y, sr, segments)
        return StructureAnalysis(segments=segments, source="songformer")
    finally:
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass
