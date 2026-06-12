# Grid-lock: align djay's beat grid and snap cues to it

## Problem

autohotcue writes cues as absolute seconds derived from beat_this's beat grid; djay
renders its own grid (stored integer BPM + an anchor it computes internally). When the
two disagree in phase, every cue looks and behaves off-grid in djay. Validated June 2026
on a 13-track test set (`/Users/payne/Music/Setlist/zzzzzTEST`).

Evidence (all empirically verified in djay's UI):

- djay's DB stores per track only `mediaItemAnalyzedData.bpm` (integer-valued F32),
  `isStraightGrid`, `keySignatureIndex` — no grid anchor.
- djay honors a grid anchor written to `mediaItemUserData.beatGridEdits` (the record it
  creates for manual grid edits). Writing `firstDownbeatPosition` moves djay's rendered
  grid; confirmed on 13 tracks.
- beat_this's reported BPM is frame-quantization-biased (reports exact frame multiples:
  120.000/125.000/157.895 where truth was 122/126/160). all-in-one-mlx has the same bias.
  djay's stored BPM was correct on every test track.
- Both models hallucinate downbeats in quiet/ambient intros (first model downbeat ~10s,
  first audible beat ~40s on one track), so "anchor at first downbeat" fails.
- Some extended mixes are spliced: constant tempo within sections but discontinuous beat
  phase across section boundaries (observed ±150 ms jumps). No straight grid fits these;
  they must be detected and skipped, not written.

## The beatGridEdits record (exact shape — do not deviate)

`tsaf.Obj` classname `ADCBeatGridEdits`, fields in this order:

| field | value |
|---|---|
| `firstDownbeatPosition` | `F32` — grid anchor, absolute seconds |
| `nrOfBeatShift` | `Marker(0x2E)` |
| `downbeatMarkers` | `Arr(0x0A, [])` |
| `firstGridSegmentTempoExponent` | `Marker(0x2E)` |
| `lastGridSegmentTempoExponent` | `Marker(0x2E)` |
| `fractionalBeatShift` | `F32` of `0.0` |

Set on the user-data root via `root.set("beatGridEdits", obj)` plus
`djaydb.ensure_cloud_key(root, "beatGridEdits")`. djay renders the grid as
`anchor + k * 60/stored_bpm` (straight grid).

## Algorithm (validated constants — keep them, expose as module constants)

New pure module `src/autohotcue/gridlock.py` (numpy in, numbers out; no DB, no file I/O;
style mirrors `cuepolicy.py`). Public entry:

```python
@dataclass(frozen=True)
class GridFit:
    bpm: float            # fitted lattice tempo (full resolution, e.g. 160.0)
    render_bpm: float     # djay's rendering tempo (e.g. 80.0) or bpm if djay's unknown
    anchor_s: float       # firstDownbeatPosition to write, in [0, render bar)
    beat_fit: float       # circular variance of energy-kept beats on the lattice
    bar_resid_std: float  # std of downbeat bar-phase residuals, in bars
    splice_jump: float    # max abs per-window phase step, in beats
    ok: bool
    reason: str           # "" when ok, else human-readable gate failure

def fit_grid(y, sr, beats, downbeats, djay_bpm: float | None) -> GridFit: ...
def snap_cues(positions: dict[str, float], fit: GridFit) -> dict[str, float]: ...
```

1. **Energy weights.** For each beat/downbeat time `t`: RMS of `y[t : t + 60/bpm_est]`
   (one beat window; one bar for downbeats). Keep entries with weight ≥ median ("kept").

2. **Tempo.** Candidate bases: `djay_bpm * k` for `k ∈ {0.5, 1, 2}` filtered to
   [60, 200], plus a regression estimate (`np.polyfit` slope over kept beats → 60/slope).
   For each base: `np.arange(base - 2, base + 2.25, 0.25)`. Score every candidate by
   circular variance `1 - |Σ w·exp(2πi·t/p)| / Σw` over kept beats (`p = 60/bpm`);
   keep the minimum, then refine ±0.25 in 0.01 steps. If djay_bpm is known and the
   winner is not within 0.1 of `djay_bpm * {0.5, 1, 2}` → gate fail
   (`reason="fitted tempo disagrees with djay's BPM"`). `render_bpm` = djay_bpm when
   known (the multiple that matched), else the winner.

3. **Gates** (any failure → `ok=False`, no grid write, cues left unsnapped):
   - `beat_fit > 0.10` → "beats do not fit a straight grid" (clean tracks measured
     0.0046–0.08; the spliced track 0.157).
   - Splice scan: split kept beats into 60 s windows (≥10 beats each); per-window
     circular-mean phase on the winning lattice; max absolute circular step between
     consecutive windows > 0.15 beats → "phase jumps between sections (spliced edit?)".

4. **Bar phase / anchor.** Weighted circular mean of kept downbeats mod the *render*
   bar period `P = 4 * 60 / render_bpm`; residual std (in bars) → `bar_resid_std`.
   If `bar_resid_std > 0.15`: fall back to rotation scoring — novelty at each lattice
   beat = RMS(bar after) − RMS(bar before); take the top 12 novelty beats; for each of
   the 4 rotations sum novelty mass landing on that rotation's "1"; pick argmax (energy
   jumps land on downbeats). Anchor = chosen bar phase projected into `[0, P)`.

5. **Cue snapping.** `snap_cues` projects each cue time onto the nearest point of the
   fitted beat lattice (`anchor + k * 60/bpm`). Slot letters and omissions unchanged.

## Integration

- `apply`: grid-lock is ON by default under ml engines (`ml`, `ml-librosa`,
  `ml-allin1`); add `--no-grid-lock`. Skipped entirely under `legacy` (no beat_this
  beats). Per track: run `fit_grid` (djay_bpm from the same lookup `bench` uses);
  when `ok`, write `beatGridEdits` and snap cues; when not, keep current behavior and
  print the gate reason. A track whose record already contains `beatGridEdits` keeps it
  unless `--force` is given (same semantics as existing cues; protects manual grid
  edits).
- `_write_one` builds the record via a new `djaydb.build_beat_grid_edits(anchor_s)`.
  All existing write-path invariants stay untouched: faithful-edit guard, djay-quit
  re-check, one timestamped backup per run, main-thread-only DB writes. `tsaf.py` needs
  **no changes** — keep it that way.
- `propose`: print one extra line per track, e.g.
  `grid: bpm=126.00 render=126 anchor=1.547s fit=0.0046 ok` or
  `grid: SKIP — phase jumps between sections (spliced edit?)`. Without a `--library`,
  djay_bpm is unknown; that's fine (candidates come from regression alone).
- `verify`: when the record has `beatGridEdits`, print
  `grid anchor: 1.547s` after the cue list.
- Workers compute `fit_grid` inside the analysis step (it only needs y/beats/downbeats);
  DB reads/writes stay on the main thread.

## Tests (new `tests/test_gridlock.py`; keep all existing tests green)

1. Synthetic lattice at 122 BPM with ±10 ms jitter, quiet hallucinated intro beats at
   wrong phase/tempo (low energy): tempo lands within 0.01 of 122 with djay_bpm=122 and
   with djay_bpm=None; anchor within 15 ms of truth; intro beats don't move the phase.
2. Half-time djay BPM (djay_bpm=80, true lattice 160): winner 160, render_bpm 80,
   anchor on the 80-BPM bar lattice.
3. Spliced synthetic (phase step of 0.3 beats at mid-track): splice gate trips, `ok=False`.
4. `beat_fit` gate: white-noise beat times → `ok=False`.
5. `snap_cues`: every output time is on the lattice; omitted slots stay omitted.
6. `build_beat_grid_edits`: build a doc containing the record, `serialize` → `parse` →
   `serialize` is byte-identical; field order and tags exactly as the table above
   (assert via parsed structure).
7. Rotation fallback: synthetic with ambiguous downbeats but a strong energy jump on a
   known "1" → anchor picks that rotation.

## Known limitations

- The splice scan needs at least two qualifying 60 s windows, so it is blind for tracks
  shorter than ~60 s (review finding, P3): a sub-0.25-beat splice there can be absorbed
  into the fitted tempo when no djay BPM is stored. Larger splices still trip `beat_fit`.
- djay double-time direction (`render_bpm > fitted`) can reach the rotation fallback;
  its anchor space is fully covered there (verified), unlike the removed half-time case.

## Out of scope (do not implement)

- Writing/changing `mediaItemAnalyzedData.bpm`.
- Multi-segment grids via `downbeatMarkers`.
- Any change to cue *placement* policy (separate workstream).
- Any change to `tsaf.py`.

## Conventions

- Python 3.13, `from __future__ import annotations`, type hints, 4-space indent,
  ~100-col lines, double-quoted strings, stdlib-first. Comments only for constraints
  the code can't show. Run `uv run pytest` before finishing — must be green.
