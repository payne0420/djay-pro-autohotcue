# 03 — SongFormer engine: bench and merge-or-drop decision

**Goal:** decide the fate of the `ml-songformer` structure engine with numbers.
It was implemented in a parallel session on the `songformer` branch (worktree)
after MOSS-Music was evaluated and rejected; it has been waiting on ground
truth (item 01) ever since.

## Current state

- Branch `songformer` (check `git branch -a` / worktrees via `git worktree
  list`) implements a SongFormer-based structure backend, presumably wired as
  `--engine ml-songformer`. Verify the branch still rebases cleanly onto
  current `main` — `main` has since gained grid-lock (`gridlock.py`,
  `analysis.TrackAnalysis.grid_fit`, cli integration), which touched
  `analysis.py` and `cli.py`.
- No bench numbers exist for it (that was the blocker).
- Context from the research that produced it: see memory/session notes
  (`moss-music-songformer-research`) — MOSS-Music was rejected; SongFormer was
  the promising structure-backend lead.

## Plan

1. Rebase `songformer` onto `main`; resolve conflicts (most likely in
   `analysis.py` engine dispatch and `cli.py` engine choices; grid-lock must
   keep working under the new engine — it only needs beat_this beats, which
   every ml engine has).
2. Add `ml-songformer` to `bench.VALID_ENGINES` if the branch predates that
   guard.
3. Run: `uv run autohotcue bench bench-data/truth-zzzzztest.json
   --engines ml,ml-allin1,ml-songformer -j 4` (note: check whether songformer
   can fan out across workers or needs the allin1-style jobs=1 clamp in
   `analysis.effective_parallel_jobs`).
4. Compare per-slot hit-rates, MAE, and runtime/track.
5. **Decision rule:** merge only if it beats the best current engine on D/E
   slots by a margin that survives the small sample (and isn't 10× slower);
   otherwise record the table in the doc and close the branch. Note that if
   item 02 (bass-detector engine) is done first, songformer must beat *that*
   to earn a merge — structure models may be obsolete for this use case.

## Acceptance criteria

- A committed bench table covering `ml`, `ml-allin1`, `ml-songformer` on the
  same truth.json.
- An explicit merge / close decision recorded here with one paragraph of
  rationale.
- If merged: docs updated (README engine list, CLAUDE.md engine flag), full
  pytest green, optional-extra dependency story documented like
  `docs/allin1-backend.md`.

## Notes

- Depends on item 01 (truth.json). Do not bench on machine-generated cues —
  that measures agreement, not quality.
- The runtime column matters: the user batch-processes 100-track playlists;
  an engine that doubles wall-clock needs a quality win to justify it.
