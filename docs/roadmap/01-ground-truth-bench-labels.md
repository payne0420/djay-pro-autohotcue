# 01 — Ground-truth bench labels (`truth.json`)

**Goal:** a hand-verified `truth.json` so `bench` can score cue placement
numerically. This is the critical path: it unblocks the placement redesign (02)
and the SongFormer comparison (03), and replaces per-iteration listening with a
per-iteration table.

## Current state

- `bench` exists and works (`src/autohotcue/bench.py`): per-slot ±1-beat /
  ±1-bar hit-rates and MAE, shared BPM yardstick per track, parallel over
  engines. It has **never run with real labels** — no `truth.json` exists.
- Expected format (`bench.load_ground_truth`):

  ```json
  {"tracks": [
    {"path": "/Users/payne/Music/Setlist/zzzzzTEST/track.opus",
     "cues": {"A": 0.45, "D": 91.2, "E": 162.5}}
  ]}
  ```

  Slots are letters A–H; only labeled slots are scored; omissions count as
  misses for labeled slots only.
- Slot semantics (djaydb.CUE_LABELS): A First Beat, B Loop In,
  C Vocal/Buildup, D Drop, E Breakdown, F Special, G Outro, H Loop Out.
- The 13-track test set `/Users/payne/Music/Setlist/zzzzzTEST` already carries
  machine cues + grid-locked anchors (June 2026), so labeling is *correcting*,
  not placing from scratch. Two of the 13 (Black Friday, AWGAZI) are spliced
  edits — label them anyway; they test robustness.

## Plan

1. **Write the dump script** (small, new): `scripts/dump_truth.py` or a
   `truth` CLI subcommand — read each track's `cuePoints` from the live
   `MediaLibrary.db` (read-only; reuse `djaydb.DjayDB` + the cue-number→letter
   mapping) and emit `truth.json`. Include only tracks under the given folder.
2. **User labels in djay**: load each zzzzzTEST track, drag every cue to where
   it *should* be (delete cues that shouldn't exist, add ones that should).
   Rough effort: 13 tracks × ~4 min.
3. **Dump** → `bench-data/truth-zzzzztest.json` (new folder; never commit the
   user's actual library DB, but truth.json itself is fine to commit).
4. **Baseline run**:
   `uv run autohotcue bench bench-data/truth-zzzzztest.json --engines ml,ml-allin1 -j 4`
   Record the table in this doc (or a results file) as the baseline the
   redesign must beat.
5. Optional second tranche: 10–15 tracks from the Top-100 playlist for genre
   coverage; same flow.

## Acceptance criteria

- `truth.json` with ≥13 tracks, every track having at least A and D labeled.
- `bench` runs clean against it for `ml` and `ml-allin1` and prints the
  per-slot table; baseline numbers recorded.
- Dump script committed with a test (synthetic DB → expected JSON).

## Notes / gotchas

- Verify cue letters map by `number` field (pad index 0–7), not by comment
  text — the user may have dragged cues without renaming comments.
- djay must not be running during the dump only if writes are involved; the
  dump is read-only and safe with djay open, but label state must be saved
  (quit djay or at least switch tracks) before dumping.
- Tracks where the user deletes a slot entirely: omit that letter from
  `cues` — do not write 0.0.
