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

Additional workaround in autohotcue: PyPI `all-in-one-mlx` 1.0.5 passes
`return_embeddings=` to single-fold `AllInOneMLX`, which rejects it. We use **harmonix-fold0**
behind a thin `_AllInOneForwardWrapper` (ensemble `harmonix-all` accepts the kwarg but is
~3× slower). Inference runs a direct forward + functional postprocess
(`_allin1._functional_result`) instead of `run_inference_mlx_spec`, skipping the metrical
DBN whose beats/downbeats autohotcue discards in favor of beat_this — segment parity with
the helper is pinned by tests (see `docs/allin1-performance.md`).

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

| Track | audio | ml (librosa) | ml-allin1 |
|-------|-------|-------------|-----------|
| O'Flynn — Kelsier | 222s | 5.7s | ~12.3s |
| Maty Owl — Green & Blue (Extended) | 396s | 8.0s | ~20.1s |

Earlier figures of 57-260s/track were measured under GPU/memory contention and
are wrong by 4-11x — see `docs/allin1-performance.md` for the clean per-phase
breakdown. ml-allin1 cost is dominated by demucs-mlx separation (~60% of
per-track cost, linear in duration); beat_this is unchanged. Subsequent tracks
in one process reuse loaded demucs/model singletons (~2.5-4s saved).

## Memory

On Apple Silicon, MLX retains freed Metal buffers in an in-process cache by design (reuse
across allocations). In a long folder run this can look like a leak: `phys_footprint` climbed
to ~25 GB (all `IOAccelerator`) while Python `malloc` stayed ~216 MB.

autohotcue caps that cache at backend init via `mlx.core.set_cache_limit` (default **3 GB**,
override with `AUTOHOTCUE_MLX_CACHE_GB`) and calls `mlx.core.clear_cache()` after each
`segment_structure_allin1` so memory returns between tracks. Loaded demucs/model weights are
unchanged; only transient GPU buffer cache is reclaimed.

The cap is load-bearing for speed, not just footprint: uncapped, the cache grows to
~29 GB on a 32 GB machine and memory pressure makes separation **5-6x slower**
(measured: Kelsier demucs 7s capped vs 40-49s uncapped). 3 GB and 6 GB run at
identical speed; below ~1 GB nothing further is gained and cap 0 costs ~30%.

Separation also runs with demucs `batch_size=1` (override via
`AUTOHOTCUE_DEMUCS_BATCH`): on M2 Max it is equal-or-faster than upstream's
default 8 at ~2.7x lower MLX peak, with byte-equivalent stems (≤1e-6, seed-fixed).
torch's MPS cache is flushed before separation and stems are dropped after the
spectrogram. Measured peaks after all of this: Kelsier ~9 GB, Green & Blue (396s)
~15 GB — now dominated by the all-in-one forward pass itself, not demucs. Expect
double-digit-GB *bursts* during an `ml-allin1` analysis of long tracks on a 32 GB
machine, but no sustained occupancy between tracks or after the run. Full
measurements: `docs/allin1-performance.md`.

## Rejected paths (errors)

1. **mlx-audio-io 1.3.9** (all-in-one-mlx default): `TypeError: Unable to convert function return value to a Python type!` in `mlx_audio_io.load()` on Python 3.13 — fixed in 1.3.10.
2. **harmonix-fold0 without wrapper**: `TypeError: AllInOneMLX.__call__() got an unexpected keyword argument 'return_embeddings'` from `helpers._run_inference_mlx_spec` (compile and non-compile paths).
3. **Loader bypass**: not attempted; upstream audio-io fix sufficient.
4. **Sidecar Python 3.12**: not implemented; not required after 1.3.10.
