# Cue overhaul spike — gate verdict

Step 0 / Step 1 gate for `docs/cue-overhaul-plan.md`. Environment: Python 3.13.3, macOS arm64, Apple Silicon (MPS available).

## Dependencies

| Package | Install | Notes |
|---------|---------|-------|
| `beat-this` | **OK** | Resolved cleanly with existing `numpy>=2.4.6` pin; no floor change needed. Pulls `torch==2.12.0`. |
| `all-in-one-mlx` | **Removed** | `uv add` succeeded on 3.13, but runtime smoke failed (see below). Not kept in `pyproject.toml`. |

First `beat_this` run downloads checkpoint `final0` (~77 MB) from JKU cloud. PyTorch hub hit an SSL cert error in this pyenv; pre-caching via `curl` unblocked the spike. Document one-time network need for end users.

## beat_this — PASS

Device: **mps** (no CPU fallback needed). Audio via repo `analysis.decode` (ffmpeg) → `Audio2Beats(dbn=False)`.

| Track | Runtime | Duration | BPM-ish | 1st downbeat | Beats | Downbeats |
|-------|---------|----------|---------|--------------|-------|-----------|
| Adam Port — Move (Extended) | 7.6s | 352.7s | 120.0 | **0.10s** | 705 | 180 |
| Adassiya — Headlights (Extended) | 2.3s | 376.1s | 120.0 | **0.06s** | 804 | 292 |
| Maty Owl — Green & Blue (Extended) | 2.8s | 395.8s | 125.0 | **0.08s** | 808 | 219 |
| O'Flynn — Kelsier | 1.4s | 221.7s | 130.4 | **5.34s** | 421 | 122 |

Model load (cold): ~0.5s after checkpoint cached. Inference ~1.4–7.6s/track on MPS (first track includes ffmpeg decode + warm-up).

First downbeats land near track start on the Afro/melodic extended mixes (sub-100 ms), which addresses the legacy kick-threshold drift on loud masters. Kelsier's 5.34s first downbeat may reflect a quiet intro — worth checking in bench ground truth.

## all-in-one-mlx — FAIL

Installed on Python 3.13.3. Smoke on **Move (Extended)** failed twice:

1. Missing `demucs-mlx[convert]` — fixed with extra install; demucs weights downloaded.
2. `mlx_audio_io.load()` raises `TypeError: Unable to convert function return value to a Python type!` on both `.opus` and ffmpeg-transcoded `.wav` — appears to be a **Python 3.13 / mlx_audio_io binding bug**, not a codec issue.

No segment labels obtained. Dependency removed per gate.

## Gate verdict

| Gate | Result |
|------|--------|
| `beat-this` works on 3.13 | **Proceed** — beat/downbeat backend |
| `all-in-one-mlx` works on 3.13 | **Fail** — use **librosa Laplacian fallback** (plan Step 2b) for structure |

**Implementation should use:** `beat_this` for `track_beats()`; **librosa spectral-clustering segmentation** for `segment_structure()` (not all-in-one-mlx).
