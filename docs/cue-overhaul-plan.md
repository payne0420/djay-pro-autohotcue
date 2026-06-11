# Overhaul cue placement: learned beat/structure analysis + eval harness

## Context

Cues are sometimes wrong: cue A ("First Beat") can land far after the real first beat, and cue G ("Outro") can land ~1s before the track ends. Root causes confirmed in `src/autohotcue/analysis.py`:

- **First beat** (`lock_grid`, analysis.py:108–117): triggers on `low-band >= 0.25 × 95th-percentile kick level`. On loud masters / quiet intros the threshold is never met early, so the anchor drifts deep into the track — and every other cue inherits the error.
- **Outro** (`propose_cues`, analysis.py:242–253): backwards scan for `total energy >= 0.6`, snapped **up** to the next 4-bar boundary then clamped `min(g_snap, n-1)` — which can be the literal last bar.
- The whole section detector is absolute thresholds (`bass_on = low >= 0.45`, breakdown `< 0.25`, outro `>= 0.6`). Brittle by construction; no test measures cue quality, so tuning is gut-feel.

The user wants a general improvement **without hardcoded rules**. Research conclusion (deep-dive, June 2026):

- **`beat_this`** (CPJKU, ISMIR 2024 "Beat This! Accurate Beat Tracking Without DBN Postprocessing") — state-of-the-art beat **and downbeat** tracker. `pip install beat-this`, PyTorch ≥ 2.0, **no madmom needed** (madmom is dead on Python 3.13, which this repo pins). API: `Audio2Beats`/`File2Beats` → `(beats, downbeats)` in seconds. This replaces the constant-BPM grid + kick threshold: cue A = first tracked downbeat.
- **`all-in-one-mlx`** (MLX port of Taejun Kim's All-In-One analyzer; MIT; Python ≥ 3.10; Apple-Silicon-native, ~6 s/track) — returns BPM, beats, downbeats, and **labeled segments** (intro/verse/chorus/bridge/break/inst/solo/outro). Original `allin1` is unusable here (requires madmom + natten). Python 3.13 support unverified → gated by a spike.
- Fallback if the spike fails: **librosa Laplacian spectral-clustering segmentation** (McFee–Ellis) on beat-synchronous features, with segments labeled by per-track *relative* energy rank — still no absolute constants.

Strategy: learned models produce downbeats + structure; a **pure cue policy** maps structure → the 8-cue layout using only structural relations and per-track relative statistics; a **bench harness** scores both engines against hand-verified ground truth so future changes are measured, not guessed. Legacy DSP path kept behind `--engine legacy`. TSAF/djaydb write path untouched.

## Step 0 — Spike (gate, before any code)

1. `uv add beat-this` — must resolve on Python 3.13.3 alongside `numpy>=2.4.6`/`librosa>=0.11` (relax the numpy floor if needed; torch ≥ 2.5 has py3.13 macOS-arm64 wheels).
2. `uv add all-in-one-mlx` — may fail on 3.13; that's the gate.
3. Smoke-run both on 2–3 real tracks (incl. one `.opus`, one known-bad first-beat track): confirm first downbeat lands on the audible first beat; note runtime; try `device="mps"` once.

**Gate:** `beat-this` works → proceed (hard requirement). `all-in-one-mlx` works → it's the structure backend. It fails → implement the librosa fallback (Step 2b) and skip the dep. First use downloads model checkpoints (document one-time network need).

## Step 1 — `pyproject.toml`

Add `beat-this`, and `all-in-one-mlx` (if gate passed). Ground truth file is JSON → no YAML dep. `uv sync`, commit `uv.lock`.

## Step 2 — New `src/autohotcue/backends.py`

```python
@dataclass(frozen=True)
class BeatAnalysis:
    bpm: float                 # 60 / median(diff(beats))
    beats: np.ndarray          # seconds
    downbeats: np.ndarray
    duration_s: float
    source: str
    def bar_s(self) -> float                       # median downbeat interval — no 4/4 assumption
    def nearest_downbeat(self, t: float) -> float

@dataclass(frozen=True)
class Segment:
    start: float; end: float; label: str
    energy_rank: float = 0.0   # per-track percentile rank of mean low-band energy

@dataclass(frozen=True)
class StructureAnalysis:
    segments: list[Segment]; source: str

def track_beats(y, sr=SR, device=None) -> BeatAnalysis      # beat_this Audio2Beats, dbn=False
def segment_structure(path, y, sr, beat) -> StructureAnalysis
def bpm_octave_ratio(a, b) -> float                          # octave-normalized deviation
def init_worker(jobs) -> None                                # caps torch threads per worker
```

- Feed the **existing ffmpeg-decoded array** (`analysis.decode`) into `Audio2Beats` so `.opus` stays ffmpeg's job. allin1-mlx gets the file path (demucs stage); temp-wav transcode if it rejects the codec.
- **Lazy module-global model singletons** — macOS spawn means each `-j` worker loads the model once, reuses across tasks. Workers always `cpu`; `mps` only when single-job.
- `energy_rank` reuses `analysis.band_energy` low band → mean per segment → rank within the track. Relative only.

### Step 2b — librosa fallback (only if allin1 gate fails)

Same `segment_structure` signature: beat-synchronous CQT+MFCC → recurrence matrix → Laplacian spectral clustering (librosa `plot_segmentation` recipe), k from eigengap; boundaries snapped to downbeats; labels by rank quantiles of this track's own segments (top tier → `chorus`, bottom → `break`, first/last → `intro`/`outro` if below median rank).

## Step 3 — New `src/autohotcue/cuepolicy.py` (pure; no audio, no I/O)

`propose_cues(beat, structure, djay_bpm=None) -> CueProposal`. Normalize segments first (merge < 1 bar into the next, merge same-label neighbors). Role sets are vocabulary, not thresholds: HIGH = {chorus, solo}, LOW = {break, bridge}, EDGE = {intro, outro}. If no HIGH label exists, the non-edge segment with max `energy_rank` is HIGH; analogous for LOW. Every cue snaps to the nearest tracked downbeat.

| Pad | Rule | Fallback | Omit when |
|-----|------|----------|-----------|
| A First Beat | `downbeats[0]` | `beats[0]`, then 0.0 | never |
| B Loop In | end of intro segment (first non-intro boundary) | `= A` | never |
| C Buildup | start of segment immediately before seg(D) | — | no D / nothing between B and D |
| D Drop | start of first HIGH segment after A | max-energy_rank non-edge segment | < 2 non-edge segments (note "no drop detected") |
| E Breakdown | start of first LOW segment after seg(D) | min-rank segment strictly between D and outro | no D / no candidate |
| F 2nd Drop | start of first HIGH segment after seg(E) | second HIGH segment after D when no E | no candidate |
| G Outro | start of last outro-labeled segment | start of final normalized segment | see guard |
| H Loop Out | `= G` | | G omitted |

**Outro guard (kills the 1s-before-end bug):** G must have ≥ 8 tracked beats of audio after it; otherwise walk back one segment boundary, else last downbeat satisfying the condition. Expressed in beats, not seconds.

**Invariants:** `A ≤ B ≤ C < D < E < F ≤ G = H`; violations → **omit with a note, never clamp**. Short tracks (< 8 downbeats or < 2 segments) → A/B only, same UX as today.

## Step 4 — `analysis.py` rework

Keep `decode`/`band_energy`/`lock_grid`/`bar_profile` and rename current `propose_cues` → `propose_cues_legacy`. Add:

```python
@dataclass
class TrackAnalysis:   # consumed by cli + viz for both engines
    bpm: float; first_beat_s: float; duration_s: float; engine: str
    beats: np.ndarray | None; downbeats: np.ndarray | None
    segments: list[Segment] | None; djay_bpm: float | None

def analyze(path, known_bpm, engine="ml", device=None) -> tuple[TrackAnalysis, CueProposal]
```

djay's BPM is no longer used for placement under ml — only **cross-checked** (`bpm_octave_ratio` > 2% → note "djay says 124.0, tracked 126.1"). Never a gate; `apply` writes absolute seconds.

## Step 5 — `cli.py`

- `--engine {ml,legacy}` (default `ml`) on propose/viz/apply/bench.
- Call sites (cli.py:99, 114, 129, 223, 278) switch to `analysis.analyze(...)` — module-level, picklable, drop-in for the existing `ex.submit` pattern.
- Both pools get `initializer=backends.init_worker, initargs=(jobs,)`; device `mps`-if-available only when effective jobs == 1.
- New `bench` subcommand → `bench.cmd_bench`.
- **DB write path untouched** (`_write_one`, backups, djay-quit re-check, main-thread-only writes).

## Step 6 — `viz.py`

`render` takes `TrackAnalysis`: add translucent labeled segment spans + downbeat tick marks. Makes ground-truth labeling a read-off-the-PNG task.

## Step 7 — New `src/autohotcue/bench.py`

Ground truth JSON: `{"tracks": [{"path": "...", "cues": {"A": 0.512, "D": 92.31, ...}}]}` — partial slots fine; owner labels ~15–30 tracks. `autohotcue bench truth.json --engines ml,legacy -j 4` prints per-slot table: n, hit-rate within ±1 beat, within ±1 bar, MAE, runtime/track — same BPM yardstick for both engines (djay BPM via `_bpm_for` when `--library` given, else tracked).

## Step 8 — Tests

- `tests/test_cuepolicy.py`: synthetic `mk_beat(bpm, bars, meter)` / `mk_segs(spec)` builders, no audio. Cases: canonical EDM shape → all 8 cues on downbeats + ordering invariant; **outro-guard regression test** (last segment ends at EOF → G still has ≥ 8 beats remaining); no-HIGH-label resolved by rank; uniform ranks → D/E/F omitted with notes; sub-bar segment merged; intro-less → B == A; 3/4 meter works; violation → omission not clamp.
- `tests/test_bench.py`: truth parsing + metric math.
- `tests/test_backends.py`: `bpm_octave_ratio` math; model tests behind `pytest.importorskip` + skip-marker (same pattern as the library round-trip test).

## Step 9 — Docs

AGENTS.md + README: new deps (torch, first-run checkpoint download), module map, `--engine`, bench workflow, parallelism rules (model singleton per worker, cpu-only when `-j > 1`, DB writes main-thread).

## Risks

| Risk | Mitigation |
|---|---|
| beat-this/torch vs numpy 2.4 pin on py3.13 | Spike gates; relax numpy floor or install from git |
| all-in-one-mlx broken on 3.13 | Gate + fully specified librosa fallback (2b), same interface |
| `.opus` rejected by model loaders | Beat backend eats our ffmpeg array; allin1 gets temp-wav |
| MPS + multiprocessing | Workers forced cpu; mps single-job only |
| Speed regression | Singleton per worker; bench reports runtime/track |
| Model downbeats disagree with djay's displayed grid | Cues are absolute seconds (non-fatal); BPM cross-check note per track |

## Verification

```bash
uv sync && uv run pytest                                    # incl. TSAF round-trip, must stay green
uv run autohotcue propose "/path/track.opus"                # ml engine + notes
uv run autohotcue propose "/path/track.opus" --engine legacy
uv run autohotcue viz "/path/track.opus" map.png            # segments + downbeat ticks
uv run autohotcue propose "/path/folder/" -j 4              # parallel ml
uv run autohotcue bench truth.json --engines ml,legacy      # A ±1-beat and G ±1-bar hit-rates must beat legacy
# write path unchanged — test against a COPY of the library:
uv run autohotcue apply track.opus --library /tmp/libcopy/MediaLibrary.db
uv run autohotcue verify track.opus --library /tmp/libcopy/MediaLibrary.db
```

Acceptance: on the labeled set, cue A (±1 beat) and cue G (±1 bar) hit-rates materially beat legacy; `apply`/`verify` unchanged; legacy engine output identical to today.

## Research sources

- [Beat This! (CPJKU/beat_this)](https://github.com/CPJKU/beat_this) — ISMIR 2024 beat+downbeat tracker
- [All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one) / [paper](https://arxiv.org/abs/2307.16425)
- [all-in-one-mlx (Apple Silicon port)](https://github.com/ssmall256/all-in-one-mlx)
- [madmom py3.13 breakage](https://github.com/CPJKU/madmom/issues/478), [beat_this #9](https://github.com/CPJKU/beat_this/issues/9)
- [librosa Laplacian segmentation example](https://librosa.org/doc/main/auto_examples/plot_segmentation.html) (McFee–Ellis)
- [BeatFM / SOTA comparison](https://arxiv.org/pdf/2508.09790), [open-source beat-tracker rundown](https://biff.ai/a-rundown-of-open-source-beat-detection-models/)
