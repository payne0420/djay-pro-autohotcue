# 06 — Half-time parity: resolve with kick evidence instead of refusing

**Goal:** reduce false refusals of the "ambiguous half-time bar phase" gate by
adding an independent evidence source (kick-band onsets) for the parity
decision — without ever reintroducing guess-on-thin-evidence (the bug class
that took four review rounds to kill; see grid-lock history).

## Current state

- For half/double djay BPM (render bar spans 2 fitted bars, e.g. djay 80 /
  fitted 160), `gridlock._bar_phase` clusters kept downbeats into the two
  render-bar parities and requires the winner to carry ≥25% normalized energy
  margin (`PARITY_MARGIN`) and ≥4 members; otherwise `ok=False`
  ("ambiguous half-time bar phase") and no grid is written.
- Real-world effect: Navras (djay 80, fitted 160) was refused in the 13-track
  validation, and 1/101 in the Top-100 run. Refusal is *safe* (the track keeps
  its existing/manual anchor) but conservative — beat_this downbeat parity is
  genuinely noisy on this material, while the music itself is usually
  unambiguous.
- Why refusal was chosen: every weight-based tie-break attempted during review
  (peak weight, >2% margin branch, index parity) had a demonstrated
  wrong-anchor-with-ok=True failure. The margin gate was the safe landing.

## Plan

1. New, independent parity vote from audio (not from beat_this downbeats):
   on the *fitted* lattice, compare kick-band onset strength at the two
   candidate render-bar phase classes — in half-time perception the "1" kick
   of each render bar is typically heavier / more reinforced (sub-bass weight)
   than the mid-bar kick. Compute a per-class energy profile over the loud
   region and a normalized margin, mirroring the existing cluster margin.
2. Decision rule (strictly additive, never replaces the gate):
   - downbeat-cluster margin ≥ PARITY_MARGIN → use it (current behavior);
   - else if kick-evidence margin ≥ its own threshold AND (when the downbeat
     margin is non-trivial) both votes agree → accept that parity;
   - else → refuse exactly as today. Disagreement between the two sources is
     always a refusal.
3. Constants as module-level (`KICK_PARITY_MARGIN`, suggested start 0.25,
   tuned on real tracks).
4. Tests: synthetic half-time lattices with reinforced-"1" kicks (clear case →
   resolved), equal kicks (→ still refused), conflicting votes (→ refused),
   plus all existing parity regressions unchanged. Real-world check: Navras
   should resolve to the user-validated anchor (~1.12 s phase, June 2026);
   verify against the live DB record before/after.

## Acceptance criteria

- Navras gets `ok=True` with an anchor matching the validated manual one
  (±20 ms), via the kick vote.
- Every historical parity adversarial test still passes; genuinely ambiguous
  synthetics still refuse; a vote-conflict test exists and refuses.
- Refusal rate on the Top-100 set drops (re-run the apply log comparison) with
  zero new wrong anchors on spot-checked tracks.

## Notes

- This is quality-of-life, not correctness — current behavior is safe. If 01
  (truth.json) exists, prefer doing this after, so anchor changes can be
  spot-checked against labeled tracks.
- Resist any temptation to lower PARITY_MARGIN itself; the margin gate is the
  proven backstop. New evidence in, thresholds unchanged.
