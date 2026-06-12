# 02 — Placement engine redesign (Afro-House-native cue policy)

**Goal:** replace generic structure-model-driven cue placement with detectors
matched to the music: kick-band bass events + 16/32-bar phrase arithmetic on
the grid-locked lattice, with djcues' per-pad vocabulary as the slot spec.
Every iteration is accepted/rejected by `bench` against `truth.json` (item 01).

## Why the current engines fail

Both structure backends (librosa Laplacian in `ml`, all-in-one-mlx in
`ml-allin1`) segment by pop/EDM song form (verse/chorus contrast). Extended
Afro House — long DJ intros, gradual layering, flat dynamics — doesn't have
that form. Measured June 2026 on the 13-track set: the two engines placed the
*drop* 30–180 s apart on the same tracks (37 s vs 118 s, 42 s vs 229 s), and
the user judged all engines unusable. The taxonomy is wrong for the genre; the
fix is different *features*, not more tuning.

## Design

### Signals (all already validated in the grid-lock work)

1. **Kick-band energy per bar**: 30–150 Hz bandpass (scipy butter order 4,
   sosfilt — same filter as `gridlock.py` energy weights), RMS per bar on the
   fitted lattice. The genre's events are *defined* by this band: drop =
   bass-in, breakdown = bass-out, second drop = bass-return.
2. **The grid-locked lattice** (`gridlock.fit_grid`): trusted anchor + BPM →
   bars can be *counted*. House is arranged in 16/32-bar phrases; drops sit on
   phrase boundaries nearly without exception.
3. Optional secondary signals where cheap: full-band RMS (builds), spectral
   flux (fills/sweeps before drops), vocal-band energy for C.

### Detector sketch

- Compute per-bar kick RMS → binarize against an adaptive threshold
  (e.g. fraction of the track's loud-section median) → **bass-presence
  timeline** as bar intervals.
- Edges of that timeline = candidate events: first bass-in (D), first
  sustained bass-out after D (E), first bass-return after E (F).
- **Phrase snapping**: build the 16/32-bar phrase lattice from bar 1
  (the grid anchor). Snap each candidate edge to the nearest phrase boundary;
  flag (don't force) candidates further than 2 bars from one.
- Derived slots by arithmetic, not detection: A = first downbeat;
  B = A (loop-in convention); G = last bass-out or final phrase boundary;
  H = G. C (vocal/buildup) = vocal-band onset before D, else last phrase
  boundary before D.
- **Confidence + fallback**: each slot carries a confidence; below threshold,
  omit the slot (existing policy convention) rather than guess. For tracks
  whose grid was gate-refused (spliced — ~15% of extended mixes), fall back to
  the current structure-based policy or omit phrase snapping.

### Slot vocabulary reference (djcues, battle-tested on rekordbox)

A First Beat · B Loop In · C vocal onset / last build before drop ·
D drop (first chorus ≥20% in, in its terms) · E first breakdown after drop ·
F second dip-recovery · G outro · H Loop Out. Keep autohotcue's 8-slot layout
and `CUE_LABELS`; borrow the *semantics*, feed them from the detectors above.

## Implementation shape

- New pure module `src/autohotcue/bassline.py` (or extend `cuepolicy.py`):
  audio + GridFit in, slot candidates out. No I/O, mirror `cuepolicy.py` style.
- Wire as a new engine `ml-bass` first (don't touch `ml` until bench says so);
  promotion to default is a one-line change after the numbers win.
- Tests: synthetic tracks with known bass-in/out bars (constructable with the
  same synthesis helpers used in `tests/test_gridlock.py`); plus bench-based
  evaluation on truth.json.

## Acceptance criteria (initial targets — revisit after baseline)

- On `truth.json` (item 01): D-slot hit-rate ≥80% within ±1 bar; E ≥70%;
  A ≥95% within ±1 beat; every slot strictly better than or equal to the best
  current engine; no slot regresses below baseline.
- Full pytest green; spliced tracks degrade gracefully (no phrase snapping on
  refused grids).
- The bench table for `ml,ml-allin1,ml-bass` committed alongside the change.

## Process

Iterate detector/policy via the implement→review loop; `bench` is the gate for
every iteration. Call the user for an audition only when the table beats
baseline — ears are for final sign-off, not iteration.
