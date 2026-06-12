# 05 — Splice gate: close the sub-60s blind spot

**Goal:** make the splice detector work on tracks shorter than ~60 s, where it
is currently blind (P3 from the final grid-lock review, documented in
`docs/grid-lock-spec.md` Known limitations).

## Current state

- `gridlock._splice_jump` scans 60 s windows at 30 s hops plus one tail-flush
  window and takes the **max pairwise** circular distance between window
  phases (`SPLICE_STEP_MAX = 0.15` beats). With fewer than 2 qualifying
  windows (≥10 kept beats each) it returns 0.0 → gate silently passes.
- Demonstrated failure (review repro): 50 s track, 0.2-beat splice at 45 s,
  `djay_bpm=None` → `ok=True`, tempo absorbs the step (fitted 121.86), snapped
  cues up to ~74 ms off in the spliced region. With a stored djay BPM the case
  is *sometimes* caught by the BPM-tolerance gate — coincidentally, not by
  design.
- Bounded blast radius: only tracks in the 30–60 s range (the minimum-evidence
  gate already refuses <30 s), typically loops/edits rather than full tracks.

## Plan

1. Scale the window to the track: for duration < 2×window, use
   `window = max(MIN_SPLICE_WINDOW_S, duration/2)` with hop = window/2 (plus
   the existing tail-flush rule), so even a 40 s track gets ≥2 windows.
   Suggested `MIN_SPLICE_WINDOW_S = 15` — enough beats per window at house
   tempos (15 s @ 122 BPM ≈ 30 beats ≫ the ≥10 rule).
2. Keep `SPLICE_STEP_MAX` unchanged; smaller windows have noisier phase means,
   so verify the false-positive margin on clean jittered tracks (the existing
   ±20 ms jitter false-positive tests, shrunk to 40–75 s durations).
3. Regression tests: the 50 s/0.2-beat repro (both `djay_bpm=None` and set) →
   gate fail; clean 40 s and 50 s tracks with realistic jitter → `ok=True`;
   existing splice and clean-track tests stay green.

## Acceptance criteria

- The reviewer's 50 s repro returns `ok=False` with the splice reason.
- No new false positives across the clean-track jitter battery (61–300 s plus
  new 40–75 s cases).
- Full pytest green; constants exposed at module level like the others;
  `docs/grid-lock-spec.md` Known limitations updated to remove the entry.

## Notes

- Don't over-engineer: this is a contained windowing change inside
  `_splice_jump` + tests. If shrinking windows degrades the 60 s+ behavior in
  any way, scope the scaling strictly to `duration < 120 s`.
