# ml-bass: Afro-House-native cue placement (kick-band events + phrase arithmetic)

Implements `docs/roadmap/02-placement-engine-redesign.md`. New engine `ml-bass`:
cues come from kick-band (30–150 Hz) bass-in/bass-out/bass-return detection plus
8-bar phrase snapping on the grid-locked lattice from `gridlock.fit_grid`
(16/32-bar offsets are reported in notes for diagnostics only).
The structure backends (librosa Laplacian / all-in-one-mlx) are NOT used by this
engine — no `segment_structure` call at all.

`truth.json` does not exist yet, so there is no bench gate in this iteration.
Validation is labels-free: synthetic unit tests with known bass-in/out bars, and
a corpus sanity run done by the orchestrator (not part of this change).

## New module `src/autohotcue/bassline.py`

Pure: numpy + dataclasses in, `CueProposal` out. No I/O, no DB, no decoding.
Style mirrors `cuepolicy.py` / `gridlock.py` (constants at module top,
`from __future__ import annotations`, small functions, double quotes, ~100 cols).

Imports allowed: `from autohotcue.analysis import CueProposal` (same pattern as
`cuepolicy.py`; `analysis.analyze` imports bassline lazily, so no cycle),
`from autohotcue.backends import BeatAnalysis, bpm_octave_ratio`,
`from autohotcue.gridlock import GridFit, kick_band`.

### One-line change to `gridlock.py` (the ONLY change to that file)

Add a public wrapper, directly below `_kick_band`:

```python
def kick_band(y: np.ndarray, sr: int) -> np.ndarray:
    """Public 30-150 Hz bandpass used by both grid fitting and bassline events."""
    return _kick_band(y, sr)
```

No other change to `gridlock.py`. Do not touch its constants, gates, or fitting.

### Constants (module top, exported for tests)

```python
LOUD_REF_PCTL = 90.0       # percentile of per-bar kick RMS for loud_ref
BASS_ON_FRAC = 0.75        # ON threshold, fraction of loud_ref
BASS_OFF_FRAC = 0.55       # hysteresis OFF threshold (must be < BASS_ON_FRAC)
DROP_MIN_ON_BARS = 8       # bass-in must sustain this long to be D
PRE_DROP_MIN_OFF_BARS = 4  # bars of bass-absence required before a drop
BREAK_MIN_OFF_BARS = 8     # bass-out must sustain this long to be E
RETURN_MIN_ON_BARS = 8     # bass-return must sustain this long to be F
PHRASE_BARS = 8            # snapping lattice (16/32-bar offsets still reported)
SNAP_MAX_BARS = 2          # max distance moved by phrase snapping
A_MIN_FRAC = 0.25          # A = first bar with kick RMS >= this * loud_ref
MIN_DROP_FRAC = 0.20       # first bass-in must start this far in to count as D (not groove)
OUTRO_MIN_BARS = 2         # G must leave >= 2 bars (8 beats) before track end
MIN_BARS = 24              # below this, place A/B only (short-track convention)
```

### Public API

```python
@dataclass(frozen=True)
class BassAnalysis:
    bar_starts: np.ndarray   # bar start times, seconds
    bar_period: float        # seconds per bar (median spacing in fallback mode)
    kick_rms: np.ndarray     # per-bar 30-150 Hz RMS, same length as bar_starts
    bass_on: np.ndarray      # bool per bar, after hysteresis
    loud_ref: float          # LOUD_REF_PCTL percentile of per-bar kick RMS
    phrase_origin: int | None  # A's bar index when phrase snapping is active
    snapped: bool            # True when fit.ok and phrase snapping was applied

def analyze_bass(
    y: np.ndarray, sr: int, beat: BeatAnalysis, fit: GridFit | None
) -> BassAnalysis: ...

def propose_cues_bass(
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    fit: GridFit | None,
    djay_bpm: float | None = None,
) -> tuple[CueProposal, BassAnalysis]: ...
```

### Bar lattice

- When `fit is not None and fit.ok`: `bar_period = 4 * 60 / fit.render_bpm`;
  `bar_starts = fit.anchor_s + k * bar_period` for `k = 0, 1, ...` keeping only
  bars that fit entirely inside the track (`start + bar_period <= duration + 1e-6`,
  `duration = len(y) / sr`). Phrase snapping is ACTIVE.
- Otherwise (no fit, or gate-refused): `bar_starts = beat.downbeats` as-is;
  `bar_period = beat.bar_s()`; per-bar windows run to the next downbeat (last
  window uses `bar_period`). Phrase snapping is DISABLED and the proposal gets a
  note: `"grid not locked ({reason}); phrase snapping disabled"` (use
  `fit.reason` when fit exists, else `"no grid fit"`). Detection still runs —
  spliced/gate-refused tracks must degrade gracefully, not fail.

### Bass-presence timeline

1. `kick_rms[i]` = RMS of `kick_band(y, sr)` over bar i's window.
2. `loud_ref` = 90th percentile of per-bar kick RMS (`LOUD_REF_PCTL`); robust
   to tracks where bass-on bars are a minority of the loud bars. Guard: if
   `loud_ref <= 0`,
   place A/B at the first downbeat (cuepolicy convention), add a note, return.
3. Hysteresis state machine over bars, starting OFF:
   OFF→ON when `kick_rms[i] >= BASS_ON_FRAC * loud_ref`;
   ON→OFF when `kick_rms[i] < BASS_OFF_FRAC * loud_ref`.
   `bass_on[i]` = state after bar i.

### Events (run-length analysis of `bass_on`)

- **D (Drop)** = start bar of the first ON-run with length >= `DROP_MIN_ON_BARS`
  whose immediately preceding OFF-run has length >= `PRE_DROP_MIN_OFF_BARS`, and
  which additionally qualifies as either a **return** (at least one earlier ON-run
  existed before that OFF-run) or **late** (start bar >= `MIN_DROP_FRAC * n_bars`).
  The first bass-in that is merely the groove starting (no prior bass, not late
  enough) is never D — the real drop is the payoff/return after a bass-out.
  Omit D with note `"D (Drop): omitted (no qualifying drop; groove only)"` when
  no run qualifies (including tracks whose bass is on from the start with no
  preceding OFF-run). If a computed D would coincide with A's bar, omit D with a
  note instead (backstop; should not occur in normal detection).
- **E (Breakdown)** = start bar of the first OFF-run with length >=
  `BREAK_MIN_OFF_BARS` (8 bars) that starts after D. Short bass dips (< 8 bars)
  are skipped so the real breakdown is cued. Omit (with note) when D omitted or
  no candidate.
- **F (2nd Drop)** = start bar of the first ON-run with length >=
  `RETURN_MIN_ON_BARS` that starts after E. Omit (with note) when E omitted or
  no candidate.
- **A (First Beat)** = first bar with `kick_rms >= A_MIN_FRAC * loud_ref`
  (skips silent/pad bars in front of the lattice). **B = A** (loop-in
  convention).
- **C (Buildup)** = when D exists and snapping is active: if the OFF-run
  immediately before D begins **after** A's bar (bass was on earlier — a
  build/vocal section, not a plain intro), C is the **start bar of that OFF-run**
  (pre-drop bass-out → build). Snap it on the 8-bar lattice like D/E/F; if
  snapping would place C at or beyond D or at/before B, keep the raw bar; if
  still out of range, omit with note. When the preceding OFF-run starts at or
  before A's bar, fall back to phrase arithmetic: last phrase boundary strictly
  between B and D (normally `D - PHRASE_BARS` when D is on a boundary). Detected
  C events get the same machine-greppable note prefix as D/E/F, e.g.
  `"C: raw bar 29, off8=-3, off16=+5, off32=+5, off-phrase (unsnapped)"`.
  Omit with note when D is absent or nothing valid lies between B and D.
  When phrase snapping is disabled (gate-refused / not lattice-locked), a
  pre-drop OFF-run C is still placed at its raw bar like D/E/F — no snapping and
  no off8= notes.
- **G (Outro)** = start bar of the trailing OFF-run (bass out through the end of
  the track) when one exists with length >= `OUTRO_MIN_BARS`; else the last
  phrase boundary (when snapping active) or last bar minus `OUTRO_MIN_BARS`
  (when not) that leaves >= `OUTRO_MIN_BARS` bars before track end. G must be
  later than every placed earlier cue, else omit. **H = G**. G must also leave
  >= `OUTRO_MIN_BARS` bars before track end; a snap that violates this falls
  back to the raw bar.
- Short tracks: `len(bar_starts) < MIN_BARS` → A/B only plus note
  `"track too short for bass structure analysis; first beat only"`.

### Phrase snapping (active only when `fit.ok`)

- `phrase_origin` = A's bar index `a0`. Phrase boundaries = `a0 + k * PHRASE_BARS`
  (8-bar lattice; corpus evidence: 85% of raw drops land within 1 bar of an
  8-bar boundary vs 59% for 16-bar).
- For each detected event in {C, D, E, F, G} with raw bar index `b`: let `d` =
  signed distance to the nearest 8-bar boundary. If `|d| <= SNAP_MAX_BARS`, move
  the event to that boundary; otherwise keep `b` (flag, don't force).
- Every detected C/D/E/F gets a machine-greppable note with the RAW (pre-snap)
  bar and offsets to the nearest 8-, 16-, and 32-bar boundaries, e.g.:
  `"D: raw bar 33, off8=+1, off16=+1, off32=+1, snapped to bar 32"` or
  `"D: raw bar 37, off8=-3, off16=+5, off32=+5, off-phrase (unsnapped)"`.
  Exact prefix format
  `"{slot}: raw bar {b}, off8={d8:+d}, off16={d16:+d}, off32={d32:+d}, "` —
  the corpus statistics run greps these. `off8` is relative to `PHRASE_BARS`;
  `off16`/`off32` use fixed 16/32-bar lattices from the phrase origin.
- Positions are `bar_starts[index]` of the final (snapped) bar — already on
  djay's rendered lattice when `fit.ok`.

### Ordering / final checks

- Reuse `cuepolicy._check_monotonicity` (import it; do not duplicate). The
  detectors already yield A <= D < E < F by construction; the shared check is a
  backstop, same notes convention.
- When `djay_bpm` is given and `bpm_octave_ratio(beat.bpm, djay_bpm) > 0.02`,
  add the same `"djay says {djay:.1f}, tracked {tracked:.1f}"` note cuepolicy
  uses.

## Wiring

### `analysis.py`

- `VALID_ENGINES` gains `"ml-bass"`. `normalize_engine("ml-bass")` →
  `("ml-bass", None)` (no structure backend; update the docstring).
- `effective_parallel_jobs`: ml-bass fans out like ml (only allin1 is pinned
  to 1) — no change needed beyond the engine being valid.
- `analyze()`: new branch for `track_engine == "ml-bass"` — decode →
  `track_beats` → `fit_grid` → `propose_cues_bass(y, SR, beat, grid_fit,
  djay_bpm=known_bpm)`. No `segment_structure` call. `TrackAnalysis` gains a new
  field `bass: object | None = None` (a `BassAnalysis` for ml-bass); `segments`
  stays None for this engine.

### `cli.py`

- `_ENGINE_CHOICES` gains `"ml-bass"`; `_ML_ENGINES` gains `"ml-bass"` (grid-lock
  semantics apply to it like the other ml engines). Update the `--engine` help
  string and the module docstring's engine list.
- No other CLI behavior changes. `apply` works for ml-bass for free via the
  existing path (cues are already on the lattice; `snap_cues` is a no-op for
  on-lattice points). Do NOT touch any write-path guard, backup, or djay-quit
  logic.

### `viz.py`

- When `track.bass` is present (ml-bass), render the bass-presence timeline
  instead of segments: for each contiguous ON interval of `bass_on`, an
  `axvspan(start, end)` in `"#e6394622"`; label "bass" centered in the first
  span only. When `track.bass.snapped`, draw phrase-boundary ticks (every
  `PHRASE_BARS` bars from `phrase_origin`) as vertical lines `"#00000055"`,
  linewidth 1.2, zorder 2 — visually heavier than the downbeat ticks.
- Existing segment rendering for other engines unchanged.

### Docs

- `CLAUDE.md`: extend the engine list mentions (`--engine {ml,ml-librosa,ml-allin1,ml-bass,legacy}`)
  and add one sentence describing ml-bass (kick-band events + phrase snapping;
  not the default). Keep edits minimal.

## Out of scope (do NOT do)

- No change to `tsaf.py`, `djaydb.py`, `bench.py`, or any write-path safety
  guard. No DB writes anywhere. Do not run `apply` or `bench`.
- Do not change the default engine; `ml` stays the default everywhere.
- Do not modify `cuepolicy.py` (importing from it is fine).
- No new dependencies (scipy/numpy already present).
- Vocal-band C detection is future work — C comes from the pre-drop OFF-run when
  bass was on earlier, else phrase arithmetic between B and D.

## Tests: new `tests/test_bassline.py` (+ one e2e case)

Reuse the synthesis approach of `tests/test_gridlock.py` (`_synth_lattice`
pattern). New helper for this file:

```python
def _synth_bass_track(bpm, bars, *, true_anchor=1.5, bass_bars=(), kick=0.9,
                      bass_amp=0.4, pad_bars=(), jitter_ms=0.0)
```

- Kick: 60 Hz decaying sine burst (100 ms, amplitude 0.9) on every beat of the
  straight lattice — gives kick-only bars genuine 30–150 Hz RMS (~0.5× bass-bar
  RMS) so spec thresholds separate cleanly.
- Bass: continuous 55 Hz sine of amplitude `bass_amp` added over each bar index
  in `bass_bars` (this dominates the 30–150 Hz band RMS vs kick-only bars).
- Pad (for A tests): sum of 800/1200 Hz sines, amplitude 0.3, over `pad_bars`
  (audible but with no kick-band content).
- Returns `(y, beats, downbeats)`; beats/downbeats on the true lattice (do not
  run beat_this in unit tests). Build `BeatAnalysis` directly and get a real
  `GridFit` via `gridlock.fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)`.

Required cases (assert positions in seconds within half a bar unless stated):

1. **Full layout**: 124 BPM, 96 bars, kick from bar 0, bass on `[32, 64)` and
   `[80, 92)`. Expect A=B=bar 0, C=bar 24 (D−8), D=bar 32 (off8=0), E=bar 64,
   F=bar 80, G=H=bar 92 (trailing off-run); fit.ok true; all positions on the
   lattice (`(t - anchor) / bar_period` integral within 1e-6).
2. **Snap +1**: bass-in at bar 33 → D snapped to bar 32; note contains
   `"raw bar 33"` and `"off8=+1"`.
3. **Off-phrase flag**: bass-in at bar 37 → D stays at bar 37; note contains
   `"off-phrase"`.
4. **No transition**: bass on from bar 0 to end → D/E/F omitted with notes;
   A/B present; G at last phrase boundary; monotonic.
5. **Hysteresis**: single 1-bar dip inside the groove (one bar without bass) →
   no E at that bar (first qualifying E is a later >= 8-bar dropout or absent).
6. **Breakdown skips short dip**: 4-bar bass dropout skipped; >= 8-bar dropout
   becomes E; return after E becomes F (deterministic D/E/F bars).
7. **Gate-refused fallback**: hand-built `GridFit(ok=False, reason="phase jumps
   between sections (spliced edit?)", ...)` → proposal still produced from model
   downbeats; `snapped` False; a note contains `"phrase snapping disabled"`; no
   `off8=` notes.
8. **Pad intro**: pad on bars 0–7 (no kick), kick from bar 8, bass from bar 40
   → A=bar 8, D=bar 40 (off8=0 relative to phrase origin 8).
9. **Short track**: 12 bars → only A/B, with the short-track note.
10. **Wiring**: `normalize_engine("ml-bass") == ("ml-bass", None)`;
   `"ml-bass" in VALID_ENGINES`; `effective_parallel_jobs("ml-bass", 4) == 4`;
   `cli._ENGINE_CHOICES` contains `"ml-bass"` and `cli._is_ml_engine("ml-bass")`.
11. **Ordering**: in case 1's proposal, assert A <= B <= C < D < E < F <= G == H.

E2E (`tests/test_e2e.py`): add one ml-bass case parametrized over the existing
`E2E_TRACKS`, with the same missing-file skip guard — run
`analysis.analyze(path, engine="ml-bass")`; assert the proposal has "A", passes
the existing `_check_ordering` helper, and `track.bass is not None`. Do NOT
reuse the `_on_downbeat` assertion: ml-bass cues sit on the fitted grid-lock
lattice, not on model downbeats. Instead, when `track.grid_fit.ok`, assert each
position is on that lattice (`(t - anchor_s) / (60 / fit.bpm)` integral within
1e-6). Must skip cleanly when the track is absent (CI).

All existing tests must stay green: run `uv run pytest` (full suite) and make
it pass before finishing.

## Conventions

Python 3.13, `from __future__ import annotations`, type hints, 4-space indent,
~100-col lines, double-quoted strings, stdlib-first. Comments only for
constraints the code can't show. No formatter is configured — match the
surrounding style.
