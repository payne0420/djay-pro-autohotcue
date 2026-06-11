# ml-allin1 structure backend

Optional structure segmentation via [all-in-one-mlx](https://github.com/ssmall256/all-in-one-mlx)
on Apple Silicon. Beat/downbeat tracking stays **beat_this**; all-in-one supplies labeled
segments only (intro/verse/chorus/bridge/break/inst/solo/outro). Cue placement still snaps
to beat_this downbeats via `cuepolicy.py`.

## Chosen path: **upstream-fix**

| Investigation step | Result |
|--------------------|--------|
| 1. Newer mlx-audio-io / all-in-one-mlx | **PASS** — `mlx-audio-io>=1.3.10` fixes Python 3.13 `TypeError: Unable to convert function return value to a Python type!` (broken in 1.3.8–1.3.9). |
| 2. Loader bypass (ffmpeg array → demucs) | Not needed once 1.3.10 is pinned. |
| 3. Local patched install | Not needed. |
| 4. Python 3.12 sidecar | Not needed. |

Additional workaround in autohotcue: PyPI `all-in-one-mlx` 1.0.5 `helpers.py` passes
`return_embeddings=` to single-fold `AllInOneMLX`, which rejects it. We use **harmonix-fold0**
with a thin `_AllInOneForwardWrapper` and `compile_forward=False` (ensemble `harmonix-all`
accepts the kwarg but is ~3× slower).

## Prerequisites

- macOS Apple Silicon (MLX)
- `uv sync --extra allin1`
- MLX weights for harmonix-fold0 in one of:
  - `$ALLIN1_MLX_WEIGHTS_DIR`
  - `~/.cache/autohotcue/mlx-weights/` (auto-populated from clone on first use)
  - `./all-in-one-mlx/mlx-weights/` (local clone; gitignored)
- `ffmpeg` on `PATH`
- First demucs run downloads htdemucs weights (one-time network)

## Usage

```bash
uv sync --extra allin1
uv run autohotcue propose "/path/to/track.opus" --engine ml-allin1
```

Engines: `ml` / `ml-librosa` (librosa Laplacian, default), `ml-allin1`, `legacy`.

```bash
uv run autohotcue bench truth.json --engines ml,ml-allin1,legacy
```

## Runtime (this machine, Python 3.13.3, MPS beat_this, mlx-allin1 cold per process)

| Track | ml (librosa) | ml-allin1 |
|-------|-------------|-----------|
| Adam Port — Move (Extended) | 8.0s | 80.2s |
| Adassiya — Headlights (Extended) | 7.6s | 130.8s |
| Maty Owl — Green & Blue (Extended) | 8.0s | 260.4s |
| O'Flynn — Kelsier | 5.7s | 57.3s |

ml-allin1 cost is dominated by demucs-mlx separation + harmonix-fold0 inference; beat_this
is unchanged. Subsequent tracks in one process reuse loaded demucs/model singletons.

## Rejected paths (errors)

1. **mlx-audio-io 1.3.9** (all-in-one-mlx default): `TypeError: Unable to convert function return value to a Python type!` in `mlx_audio_io.load()` on Python 3.13 — fixed in 1.3.10.
2. **harmonix-fold0 without wrapper**: `TypeError: AllInOneMLX.__call__() got an unexpected keyword argument 'return_embeddings'` from `helpers._run_inference_mlx_spec` (compile and non-compile paths).
3. **Loader bypass**: not attempted; upstream audio-io fix sufficient.
4. **Sidecar Python 3.12**: not implemented; not required after 1.3.10.
