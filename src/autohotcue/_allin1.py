"""Optional all-in-one-mlx structure backend (Apple Silicon + mlx-audio-io >= 1.3.10)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

from autohotcue.analysis import HOP, SR, band_energy
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis

ALLIN1_LABELS = frozenset(
    {"intro", "verse", "chorus", "bridge", "break", "inst", "solo", "outro"}
)
# HARMONIX fold0 emits start/end at track edges; cuepolicy EDGE = {intro, outro}.
_EDGE_LABEL_MAP = {"start": "intro", "end": "outro"}
_WEIGHTS_MARKER = "harmonix-fold0_mlx.npz"
_MODEL_NAME = "harmonix-fold0"

_separator = None
_model = None
_weights_dir: Path | None = None


class _AllInOneForwardWrapper:
    """PyPI all-in-one-mlx 1.0.5 helpers pass return_embeddings= to single-fold models."""

    def __init__(self, model: object) -> None:
        self._model = model
        self.cfg = model.cfg  # type: ignore[attr-defined]

    def __call__(self, x, return_embeddings: bool = False):  # noqa: ARG002
        return self._model(x)


def _import_error(exc: BaseException | None = None) -> RuntimeError:
    msg = (
        "ml-allin1 requires Apple Silicon macOS with optional deps installed:\n"
        "  uv sync --extra allin1\n"
        "Also set ALLIN1_MLX_WEIGHTS_DIR to a directory containing "
        f"{_WEIGHTS_MARKER} (and sibling fold weights), or clone all-in-one-mlx "
        "with its mlx-weights/ folder."
    )
    if exc is not None:
        return RuntimeError(f"{msg}\n({type(exc).__name__}: {exc})")
    return RuntimeError(msg)


def _candidate_weight_dirs() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("ALLIN1_MLX_WEIGHTS_DIR")
    if env:
        out.append(Path(env).expanduser())
    out.append(Path.home() / ".cache/autohotcue/mlx-weights")
    out.append(Path("all-in-one-mlx/mlx-weights"))
    here = Path(__file__).resolve()
    for parent in here.parents:
        clone = parent / "all-in-one-mlx/mlx-weights"
        if clone.is_dir():
            out.append(clone)
            break
    return out


def _has_weights(path: Path) -> bool:
    return (path / _WEIGHTS_MARKER).is_file()


def resolve_weights_dir() -> Path:
    """Return a directory containing harmonix-fold0_mlx.npz."""
    env = os.environ.get("ALLIN1_MLX_WEIGHTS_DIR")
    if env:
        env_path = Path(env).expanduser().resolve()
        if _has_weights(env_path):
            return env_path

    cache = Path.home() / ".cache/autohotcue/mlx-weights"
    if _has_weights(cache):
        return cache.resolve()
    for candidate in _candidate_weight_dirs():
        resolved = candidate.expanduser().resolve()
        if resolved == cache.resolve():
            continue
        if env and resolved == Path(env).expanduser().resolve():
            continue
        if _has_weights(candidate):
            cache.parent.mkdir(parents=True, exist_ok=True)
            if not cache.is_dir():
                shutil.copytree(candidate, cache)
            else:
                for item in candidate.iterdir():
                    dest = cache / item.name
                    if item.is_file() and not dest.exists():
                        shutil.copy2(item, dest)
            return cache.resolve()
    raise _import_error()


def _ensure_mlx_env() -> None:
    os.environ.setdefault("NATTEN_MLX", "1")
    os.environ.setdefault("NATTEN_MLX_BACKEND", "metal")
    os.environ.setdefault("NATTEN_MLX_COMPILE", "1")


def _check_platform() -> None:
    if sys.platform != "darwin":
        raise _import_error()
    try:
        import mlx  # noqa: F401
    except ImportError as exc:
        raise _import_error(exc) from exc


def _get_separator():
    global _separator
    if _separator is None:
        _check_platform()
        try:
            from demucs_mlx.api import Separator
        except ImportError as exc:
            raise _import_error(exc) from exc
        _ensure_mlx_env()
        _separator = Separator(model="htdemucs", progress=False)
    return _separator


def _get_model(weights_dir: Path):
    global _model, _weights_dir
    if _model is None or _weights_dir != weights_dir:
        _check_platform()
        try:
            from allin1_mlx.models.loaders_mlx import load_pretrained_model_mlx
        except ImportError as exc:
            raise _import_error(exc) from exc
        raw = load_pretrained_model_mlx(
            model_name=_MODEL_NAME,
            weights_dir=weights_dir,
        )
        _model = _AllInOneForwardWrapper(raw)
        _weights_dir = weights_dir
    return _model


def _energy_ranks_for_segments(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
) -> list[Segment]:
    if not segments:
        return segments
    low, _, _, _ = band_energy(y, sr, HOP)
    energies: list[float] = []
    for seg in segments:
        i0 = int(seg.start * sr / HOP)
        i1 = max(i0 + 1, int(seg.end * sr / HOP))
        energies.append(float(low[i0:i1].mean()) if i1 <= len(low) else 0.0)
    from scipy.stats import rankdata

    ranks = (rankdata(energies, method="average") - 1) / max(1, len(energies) - 1)
    return [
        Segment(seg.start, seg.end, seg.label, energy_rank=float(r))
        for seg, r in zip(segments, ranks)
    ]


def _map_allin1_segments(result, duration_s: float) -> list[Segment]:
    segments: list[Segment] = []
    for seg in result.segments:
        label = seg.label.lower()
        label = _EDGE_LABEL_MAP.get(label, label)
        if label not in ALLIN1_LABELS:
            raise ValueError(
                f"unknown all-in-one segment label {seg.label!r} "
                f"(expected one of {sorted(ALLIN1_LABELS)})"
            )
        start = float(seg.start)
        end = float(seg.end)
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"non-finite segment boundary for {seg.label!r}: "
                f"start={seg.start!r}, end={seg.end!r}"
            )
        start = max(0.0, start)
        end = min(duration_s, end)
        if end > start:
            segments.append(Segment(start=start, end=end, label=label))
    return segments


def segment_structure_allin1(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
) -> StructureAnalysis:
    """Structure segments from all-in-one-mlx (demucs + harmonix-fold0)."""
    # demucs reads *path*; y/sr are the ffmpeg decode used for energy ranks
    weights_dir = resolve_weights_dir()
    try:
        from allin1_mlx.helpers import run_inference_mlx_spec
        from allin1_mlx.spectrogram import spectrogram_from_stems
    except ImportError as exc:
        raise _import_error(exc) from exc

    _ensure_mlx_env()
    separator = _get_separator()
    model = _get_model(weights_dir)
    sep_sr = separator.samplerate
    model_sr = model.cfg.sample_rate
    if sep_sr != model_sr:
        raise RuntimeError(
            f"demucs samplerate ({sep_sr}) != all-in-one model sample_rate "
            f"({model_sr}); segment timestamps would be mis-scaled"
        )
    _, stems = separator.separate_audio_file(path, return_mx=True)
    spec = spectrogram_from_stems(
        stems,
        sample_rate=sep_sr,
        backend="mlx_fast",
        return_mx=True,
    )
    result = run_inference_mlx_spec(
        path=Path(path),
        spec=spec,
        model=model,
        include_activations=False,
        include_embeddings=False,
        compile_forward=False,
    )
    segments = _map_allin1_segments(result, beat.duration_s)
    segments = _energy_ranks_for_segments(y, sr, segments)
    return StructureAnalysis(segments=segments, source="allin1")
