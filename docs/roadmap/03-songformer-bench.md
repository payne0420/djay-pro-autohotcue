# 03 — SongFormer engine: bench and keep-or-retire decision

**Goal:** judge the `ml-songformer` structure engine with numbers. The engine
is **already merged to `main`** (June 2026) and fully wired — what's missing is
the quantitative comparison against `ml` and `ml-allin1` on real ground truth.

## Current state (post-merge)

- `--engine ml-songformer` works end-to-end on `main`: SongFormer structure
  (HF `ASLP-lab/SongFormer`, pinned revision), beat_this beats, grid-lock
  snapping, cue policy unchanged. See `docs/songformer-backend.md` for the
  full backend story (install, env knobs, memory).
- Wiring that the original draft of this brief asked to verify is all done:
  `bench.VALID_ENGINES` includes it, `analysis.effective_parallel_jobs`
  clamps it to jobs=1 (allin1-style), unit + opt-in e2e tests exist.
- Qualitative-only evidence so far: on Afro House test tracks the cue layouts
  are musically sensible (Drop on first chorus, Breakdown on the bridge) and
  the paper numbers favor it (HarmonixSet HR.5F 0.703 vs All-In-One 0.596,
  arXiv 2510.02797). No bench table exists — that was and remains the blocker.
- Known costs to weigh in the decision: ~110 s/track forward on CPU
  (vs ~18 s for ml-allin1) and a ~24 GB transient RSS peak per track
  (kernel-capped at 26 GB; see item 07 for the path to fixing this).

## Plan

1. Wait for item 01 (truth.json). Do not bench on machine-generated cues —
   that measures agreement, not quality.
2. Run: `uv run autohotcue bench bench-data/truth-zzzzztest.json
   --engines ml,ml-allin1,ml-songformer` (songformer self-clamps to jobs=1;
   budget ~2 min/track and ~24 GB transient RAM per track — close heavy apps,
   or run overnight).
3. Compare per-slot hit-rates, MAE, and runtime/track.
4. **Decision rule:** `ml-songformer` stays the recommended structure engine
   only if it beats the best current engine on D/E slots by a margin that
   survives the small sample. If item 02's bass-detector engine exists by
   then, songformer must beat *that*. If it loses, keep the engine (it is
   merged, optional, and zero-cost when unused) but record the table and
   demote it in README/docs from "better boundaries" to "alternative".

## Acceptance criteria

- A committed bench table covering `ml`, `ml-allin1`, `ml-songformer` on the
  same truth.json, recorded in `docs/songformer-backend.md`.
- An explicit keep-as-recommended / demote decision recorded here with one
  paragraph of rationale, including the runtime/memory cost in the weighing.

## Notes

- Depends on item 01. Independent of item 07 (memory), but if 07 lands first,
  re-measure runtime — the SDPA path should be faster as well as smaller.
- The runtime column matters: the user batch-processes 100-track playlists;
  at ~2 min/track a full playlist is a 3.5 h overnight run as it stands.
