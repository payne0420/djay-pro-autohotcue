"""Tests for analysis backends."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autohotcue.backends import bpm_octave_ratio

_BEAT_THIS_CKPT = Path.home() / ".cache/torch/hub/checkpoints/beat_this-final0.ckpt"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (120.0, 120.0, 0.0),
        (120.0, 121.0, 1.0 / 120.0),
        (120.0, 60.0, 0.0),
        (120.0, 240.0, 0.0),
        (128.0, 64.5, abs(128.0 - 64.5 * 2) / 128.0),
    ],
)
def test_bpm_octave_ratio(a: float, b: float, expected: float) -> None:
    assert bpm_octave_ratio(a, b) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_bpm_octave_ratio_invalid() -> None:
    assert bpm_octave_ratio(0.0, 120.0) == float("inf")


def test_track_beats_callable() -> None:
    pytest.importorskip("beat_this.inference")
    from autohotcue.backends import track_beats

    assert callable(track_beats)


@pytest.mark.skipif(
    not _BEAT_THIS_CKPT.is_file(),
    reason="beat_this checkpoint not cached (download once online)",
)
def test_track_beats_on_click(tmp_path) -> None:
    """Run beat_this on a synthetic click track when checkpoint is available."""
    pytest.importorskip("beat_this.inference")
    torch = pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    from autohotcue.analysis import decode
    from autohotcue.backends import track_beats

    import soundfile as sf

    sr = 44100
    bpm = 120.0
    beat_s = 60.0 / bpm
    duration_s = 16.0
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    y_write = np.zeros_like(t)
    for beat_t in np.arange(0.0, duration_s, beat_s):
        idx = int(beat_t * sr)
        if idx < len(y_write):
            y_write[idx : idx + 400] = 0.8
    path = tmp_path / "click.wav"
    sf.write(path, y_write, sr)
    y = decode(str(path), sr=sr)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    beat = track_beats(y, sr=sr, device=device)
    assert beat.source == "beat_this"
    assert beat.duration_s == pytest.approx(duration_s, abs=0.2)
    assert len(beat.beats) >= 1
    assert beat.bpm == pytest.approx(bpm, rel=0.15)


def test_energy_rank_ties_are_uniform() -> None:
    from scipy.stats import rankdata

    energies = [1.0, 1.0, 1.0, 1.0]
    ranks = (rankdata(energies, method="average") - 1) / max(1, len(energies) - 1)
    assert all(r == pytest.approx(0.5) for r in ranks)


def test_segment_structure_empty_beats() -> None:
    from autohotcue.backends import BeatAnalysis, segment_structure_librosa

    beat = BeatAnalysis(
        bpm=120.0,
        beats=np.array([0.0, 0.5]),
        downbeats=np.array([0.0]),
        duration_s=1.0,
        source="test",
    )
    y = np.zeros(44100, dtype=np.float32)
    out = segment_structure_librosa("dummy.opus", y, 44100, beat)
    assert out.segments == []
    assert out.source == "librosa"
