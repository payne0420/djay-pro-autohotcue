# mlx-audio-io Python 3.13 binding bug (upstream draft)

**Status:** Fixed in mlx-audio-io **1.3.10** (2026-03). This note documents the original
failure for reproducibility; do not post until verified on a clean 3.13 env without the fix.

## Minimal repro

Environment: Python 3.13.3, macOS arm64, `mlx-audio-io==1.3.9` (pulled by all-in-one-mlx 1.0.5).

```python
from mlx_audio_io import load

# Either path fails identically:
load("/path/to/track.opus", mono=True, dtype="float32")
load("/path/to/transcoded.wav", mono=True, dtype="float32")
```

Observed:

```
TypeError: Unable to convert function return value to a Python type! The signature was
    load(path: str, sr: int | None = None, offset: float = 0.0, duration: float | None = None,
         mono: bool = False, layout: str = 'channels_last', dtype: str = 'float32',
         resample_quality: str = 'default') -> tuple[mlx::core::array, int]
```

Same error from `demucs_mlx.api.Separator.separate_audio_file()` which calls `mlx_audio_io.load`
internally.

## Fix

Upgrade to `mlx-audio-io>=1.3.10`. Verified on Python 3.13.3 / macOS arm64: both opus and wav
load return `(mlx.core.array, sample_rate)` successfully.

## Draft issue text (mlx-audio-io)

**Title:** Python 3.13: `load()` raises "Unable to convert function return value to a Python type"

**Body:**

On CPython 3.13.3 (macOS arm64), `mlx_audio_io.load()` fails when returning `(mx.array, int)`
from the native binding. Affected versions observed: 1.3.8, 1.3.9. Fixed in 1.3.10.

Repro:

```python
from mlx_audio_io import load
load("test.wav", mono=True, dtype="float32")
```

Expected: `(mlx.core.array, int)`
Actual: `TypeError: Unable to convert function return value to a Python type!`

Workaround: upgrade to 1.3.10+.
