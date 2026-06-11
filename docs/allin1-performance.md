# ml-allin1 performance investigation

Why did `--engine ml-allin1` look 10-40x slower than all-in-one-mlx's README claim
(5.96s/file)? Measured 2026-06-11 on Apple M2 Max, 32 GB, macOS Darwin 25.5,
Python 3.13.3, mlx 0.31.0, all-in-one-mlx 1.0.5, demucs-mlx 1.4.3. Method: fresh
process per configuration, one warmup run first, phases timed with
`time.perf_counter`, peak memory polled with `/usr/bin/footprint` (RSS cannot see
MLX/Metal allocations).

## TL;DR

**The previously recorded numbers (57-260s/track) were contaminated measurements,
not the pipeline's real cost.** Re-measured cleanly, Kelsier (222s of audio) takes
**14.8s** end-to-end and Green & Blue (396s) takes **23.6s** — and the dominant
prior-run contaminant is reproducible: letting the MLX Metal buffer cache grow
uncapped (or sharing the GPU with another MLX workload) pushes this 32 GB machine
into memory pressure and makes demucs separation **5-6x slower**. autohotcue's
`AUTOHOTCUE_MLX_CACHE_GB` cap is the fix, not a suspect.

On the same file and machine, upstream's own `allin1-mlx` CLI at default settings
takes **40.9s** — our integration is **2.7x faster** than upstream, because we cap
the cache (upstream does not) and run the single `harmonix-fold0` model instead of
the 8-fold `harmonix-all` ensemble.

## Per-phase breakdown (clean, cap 6 GB, fresh process)

Kelsier = 221.7s of audio; Green & Blue = 395.8s. "warm" = second analysis in the
same process (folder-run marginal cost).

| Phase | Kelsier | Green & Blue | Notes |
|---|---|---|---|
| import numpy+torch+mlx | 0.7-1.0s | 0.6s | fixed per process |
| import autohotcue + allin1 modules | 1.2-2.4s | 1.2s | fixed per process |
| beat_this model load (MPS) | 0.43-0.48s | 0.43s | fixed per process |
| demucs Separator init + fold0 load | 0.10-0.15s | 0.09s | fixed per process |
| ffmpeg decode (mono 44.1k) | 0.62-0.65s | 1.05s | per track |
| beat_this tracking | 0.95-1.11s | 1.68s | per track; not in upstream's pipeline at all |
| **demucs separation** | **7.0-8.3s** | **12.9-13.9s** | per track; the dominant cost |
| spectrogram (mlx_fast) | 0.53s cold / 0.22s warm | 2.11s cold / 0.27s warm | cold includes one-time `spec_fast_guard` parity check |
| allin1 nn forward (fold0) | 0.31-0.46s | 0.55-0.83s | per track |
| metrical DBN postprocess | 1.34-1.56s | 2.62-2.73s | per track; **discarded** — autohotcue uses beat_this beats |
| functional postprocess | ~0.01s | ~0.02s | the only inference output we keep |
| energy ranks (band_energy STFT) | 0.23-0.52s | 0.38-0.69s | per track, autohotcue addition |
| cue policy | <0.01s | <0.01s | |
| **CLI total (`propose`)** | **14.8s** | **23.6s** | vs 57.3s / 260.4s previously recorded |

Fixed startup is ~2.5-4s; marginal per-track cost is ~10.5-13s for a 222s track
(~20x real-time). Demucs separation is ~60% of the marginal cost.

## Upstream-claim autopsy (suspect 2)

The README's 5.96s is "one run, one file" on an M4 Max with 128 GB, macOS 26.3 —
file length unstated. Upstream's own timing reports
(`all-in-one-mlx/claude-reports/FINAL_OPTIMIZATION_REPORT.md`) break their ~6s
down as: demix 4.08s + spectrogram 0.41s + nn 0.027s + postprocess 1.83s
(+ model load 0.27s). So the claim covers **demix → spectrogram → nn →
postprocess only**. It excludes interpreter/torch imports, audio decode for
anything else, and — most importantly for comparing against autohotcue — it
contains **no beat_this pass** (their beats come from the DBN postprocess, which
we discard and replace with beat_this). Their nn figure of 27ms for an 8-fold
ensemble is ~90x faster than the same ensemble measured here (2.5s), which is far
beyond any M4-vs-M2 gap — consistent with their benchmark file being much shorter
than a 4-7-minute extended mix (the file is named `/tmp/f.wav` in their reports;
duration never stated).

Our previously recorded numbers, by contrast, were cold full-CLI walls (imports +
beat_this + decode + the whole allin1 path) — and contaminated (below). The
"10-40x" framing compared two different quantities, one of them mismeasured.

## Suspect-by-suspect results

### 1. `compile_forward=False` — exonerated (kept `False`, conflict fixed anyway)

The actual conflict: upstream's compile path captures `model.state` inside
`mx.compile(inputs=[model.state])`, and `_AllInOneForwardWrapper` did not expose
`state` → `AttributeError` (verified). Fixed by delegating `state` to the wrapped
model (`_allin1.py`), so compile can be enabled for experiments. Measured with the
fix (Kelsier, fresh process each):

| Config | nn first track | nn repeat (same spec shape) |
|---|---|---|
| compile_forward=False | 0.42-0.46s | 0.31s |
| compile_forward=True | 1.10s (trace overhead) | 0.31s |

Compile only pays back on a cache hit, and `mx.compile` keys on input shape —
every track has a different spectrogram length, so folder runs would re-trace
**every track** for a ~0.7s penalty and zero steady-state gain. Upstream agrees:
their `analyze()` defaults `mlx_compile = len(paths) > 1` precisely because
"for one-off single-track runs, compile overhead often dominates". `False` stays.

### 2. Benchmark scope mismatch — confirmed (see autopsy above)

### 3. MLX cache cap — the cap is the cure, not the disease

Kelsier, two analyses per process, peak phys_footprint via `footprint`:

| `AUTOHOTCUE_MLX_CACHE_GB` | demucs (run1/run2) | process total | peak footprint |
|---|---|---|---|
| 3 | 7.2s / 7.1s | 23.5s | ~10 GB |
| 6 (old default) | 7.3s / 8.1s | 24.7s | ~13 GB |
| 12 | 7.6s / 9.9s | 27.6s | ~24 GB |
| uncapped (999) | **48.7s / 40.1s** | **108.3s** | ~29 GB |

Uncapped, MLX retains every freed Metal buffer; the footprint climbs to ~29 GB on
a 32 GB machine, the OS hits memory pressure, and separation runs 5-6x slower
(40s+ of it system time). This reproduces the signature of the old contaminated
numbers. Green & Blue confirms at length: cap 3 → 17 GB peak, cap 6 → 21 GB peak,
identical runtime (~13s separation). **Default lowered 6 → 3 GB**: same speed on
both track lengths, 3-4 GB less worst-case footprint.

### 4. Hardware gap — real but secondary (~1.5-2x, not 10-40x)

Apples-to-apples on this machine, same file (Kelsier), upstream's own CLI:

| Pipeline | Wall | demix | spec | nn | postprocess |
|---|---|---|---|---|---|
| upstream CLI, stock defaults (harmonix-all, uncapped) | **40.9s** | 31.2s | 1.90s | 3.42s | 2.45s |
| upstream CLI + 6 GB cache cap injected | **12.9s** | 6.97s | 0.49s | 2.48s | 1.39s |
| autohotcue ml-allin1 (fold0, capped; incl. beat_this etc.) | **14.8s** | 7.0-8.3s | 0.53s | 0.31-0.46s | 1.5-1.9s |

The M4 Max's structural advantage in the README number is less its GPU (perhaps
1.5-2x on demix throughput) than its **128 GB of RAM**, which makes uncapped MLX
caching harmless. On 32 GB the uncapped default collapses; capped, their pipeline
and ours are within ~2x of their hardware-adjusted claim.

### 5. Integration overhead — none found; we are net faster

Our demucs → spectrogram → inference call sequence is identical to upstream's
in-memory path (`analyze.py`), same `Separator(model="htdemucs")`, same
`mlx_fast` spectrogram backend. `_AllInOneForwardWrapper` adds one Python call
per forward — unmeasurable. Stock upstream with `--model harmonix-fold0` crashes
(`TypeError: ... unexpected keyword argument 'return_embeddings'`, verified),
so the wrapper is what lets us run the 1-fold model at all: nn 0.31-0.46s vs the
ensemble's 2.5-3.4s. What autohotcue adds on top of upstream — ffmpeg decode,
beat_this, energy ranks — costs ~2-3s/track and is the product (beat-accurate cue
placement), not overhead.

### 6. Track-length scaling — linear (with the cap)

Green & Blue / Kelsier duration ratio = 1.79x. Measured ratios at cap 6:
demucs 1.7-1.8x, nn 1.8x, DBN 1.9x, spectrogram (warm) 1.2x. No nonlinearity.
The old 4.5x blowup (57s → 260s) was memory pressure, not scaling.

## Applied fixes (commit `c3a6b63e9ee7e7f28ffd14ee5ba69290e8bf2f00`)

1. `_AllInOneForwardWrapper.state` property — fixes the `compile_forward=True`
   crash so the compile path is testable; default stays `False` (measured: no
   benefit, per-track retrace penalty).
2. `AUTOHOTCUE_MLX_CACHE_GB` default 6 → 3 — same runtime, 3-4 GB lower peak
   footprint on both track lengths (worst case observed: 21 GB → 17 GB on a
   396s track).
3. Tests added for both (`tests/test_allin1.py`); full suite 74 passed.

Before/after, real CLI: Kelsier 15.4s → 14.8s, i.e. unchanged within noise — the
fixes are about memory headroom and correctness, not speed. The headline change
is corrected bookkeeping: the engine was never 80-260s/track on clean runs.

## Remaining recoverable speedups (ranked by effort)

1. **Skip the metrical DBN postprocess** (~1.4s/222s track, ~2.7s/396s track —
   ~13% of marginal cost). autohotcue only consumes `result.segments`; beats/
   downbeats/BPM from allin1 are discarded in favor of beat_this. Calling the
   model forward + `postprocess_functional_structure_mlx` directly in
   `_allin1.py` instead of `run_inference_mlx_spec` removes the DBN entirely.
   Moderate, contained rewrite of one call site; needs a segments-parity test
   against the current path before adoption. Proposed, not implemented.
2. **Share the band-energy STFT** (~0.2-0.7s/track). `_energy_ranks_for_segments`
   recomputes `band_energy` over the full track; a folder run could cache it per
   track alongside the decode. Small, low risk, low payoff.
3. **`spec_fast_guard=False`** (~0.3s once per process). The guard re-computes
   the first spectrogram on the reference backend as a parity check. Keeping it
   is cheap insurance; only worth disabling if cold single-track latency matters.
4. **Not recoverable / not worth it**: demucs separation dominates and is already
   linear and capped — a faster separation model (e.g. a lighter demucs variant)
   would change quality, and reusing our mono ffmpeg decode for demucs is
   impossible (it needs stereo at 44.1 kHz; its own `mlx_audio_io` load costs
   well under a second).

## Measurement caveats

- Background apps (Spotify, browser) were running but idle-ish; GPU was otherwise
  free. Run-to-run demucs spread was ±0.5-1s.
- `footprint` peaks are 1 Hz samples of a spiky signal — treat as ±1-2 GB.
- The four-track table in `docs/allin1-backend.md` predating this report was
  recorded under GPU/memory contention and overstated cost 4-11x; it has been
  replaced with clean numbers for the two re-measured tracks.
