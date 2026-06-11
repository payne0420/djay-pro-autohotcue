# autohotcue

Automatic hot cues for **djay Pro**, written directly into djay's music
catalog. Works with any audio format ffmpeg can decode — including **.opus**
and **.ogg**, which rekordbox, Serato, and the rekordbox→djay bridges refuse.

It analyzes the audio itself (no reliance on another DJ app's phrase data),
places an 8-cue layout, and writes the cues straight into djay's
`MediaLibrary.db` — no USB export, no intermediate app.

![Move — auto hot cues](move_cuemap.png)

## Why this exists

djay Pro has no automatic hot-cue feature, and every off-the-shelf option
(Lexicon, Mixed In Key → rekordbox → djay) either skips djay or can't read
`.opus`/`.ogg`. autohotcue closes both gaps by talking to djay's database
format directly.

## How it works

1. **Decode** the track to PCM via ffmpeg (handles opus/ogg/flac/m4a/…).
2. **Track beats and downbeats** with [beat_this](https://github.com/CPJKU/beat_this)
   (ISMIR 2024; default `ml` engine). Cue A lands on the first tracked downbeat.
3. **Segment structure** via librosa Laplacian spectral clustering on
   beat-synchronous features; segments are labeled by per-track relative energy.
4. **Place cues** with a pure policy (`cuepolicy.py`) that maps structure →
   the 8-cue layout. Every cue snaps to a tracked downbeat. djay's BPM is
   cross-checked only (never used for placement under `ml`).
5. **Write cues** into djay's `mediaItemUserData` record as `ADCCuePoint`
   objects, serialized in djay's own **TSAF** binary format.

The legacy `--engine legacy` path keeps the original band-energy grid analysis
for comparison.

### The cue layout

| Pad | Cue            | Anchored on |
|-----|----------------|-------------|
| A   | First Beat     | first tracked downbeat |
| B   | Loop In        | end of intro |
| C   | Vocal / Buildup| segment before drop |
| D   | Drop           | first high-energy section |
| E   | Breakdown      | first low-energy section after drop |
| F   | Special        | second high-energy section |
| G   | Outro          | outro segment (≥8 beats of audio remain) |
| H   | Loop Out       | outro start |

## The TSAF format

djay stores per-track metadata as binary blobs in a YapDatabase-on-SQLite
file. The blobs use an undocumented object-archive format (magic `TSAF`).
`src/autohotcue/tsaf.py` is a from-scratch parser **and** serializer for it,
reverse-engineered and validated by reproducing **every blob in a real
library byte-for-byte** (11,301/11,301). That byte-exact round-trip is the
safety property that makes writing to the live database safe — the test suite
enforces it.

## Usage

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and `ffmpeg`.
First run downloads beat_this model checkpoints (one-time network).

```bash
uv sync

# Analyze and print proposed cues (no writes, no djay needed)
uv run autohotcue propose "/path/to/track.opus"

# Legacy band-energy engine (pre-overhaul behavior)
uv run autohotcue propose "/path/to/track.opus" --engine legacy

# Render a waveform + segment spans + downbeat ticks + cue map PNG
uv run autohotcue viz "/path/to/track.opus" cuemap.png

# Write cues into djay's library (QUIT djay first)
uv run autohotcue apply "/path/to/track.opus"

# Read back what djay has stored for a track
uv run autohotcue verify "/path/to/track.opus"

# Score engines against hand-labeled ground truth
uv run autohotcue bench truth.json --engines ml,legacy -j 4

# Point propose/apply/verify at a folder to process every audio file in it
# (recursive). A folder apply takes ONE backup for the whole run and skips
# problem tracks (not in djay, already has cues, ...) with a summary.
# -j N analyzes N tracks in parallel (-j 0 = one worker per CPU core);
# database writes always stay serialized on the main thread.
# Parallel workers use cpu only; mps is used when -j 1.
uv run autohotcue apply "/path/to/Afro House/" -j 4
```

The track must already be in djay's **My Collection** (so the library has a
record to attach cues to). `apply` reuses djay's analyzed BPM for cross-check
and legacy placement; pass `--bpm` to override.

### Bench ground truth

Hand-label tracks in JSON:

```json
{
  "tracks": [
    {"path": "/path/to/track.opus", "cues": {"A": 0.512, "D": 92.31, "G": 180.0}}
  ]
}
```

Partial slots are fine. `bench` reports per-slot hit-rate (±1 beat, ±1 bar),
MAE, and runtime per engine using the same BPM yardstick (djay BPM when
`--library` is given, else tracked).

## Safety

- **djay must be quit** before `apply` — it refuses if `djay Pro` is running,
  and **re-checks immediately before the write** in case djay started during
  the (slow) audio analysis.
- **Exact track matching** — the track is matched by its full `file://` source
  URL. Phantom duplicates (a leftover location record with no other metadata)
  are ignored in favor of the real catalog entry; a genuinely ambiguous match
  is refused rather than guessed, so cues never land on the wrong track.
- **Faithful-edit guard** — before modifying an existing record, the original
  blob must re-serialize byte-for-byte; if autohotcue can't reproduce it
  exactly it refuses, so it can never corrupt the record's other fields.
- **Unique, consistent backups** — each `apply` run checkpoints the WAL and
  copies `MediaLibrary.db` (+ `-wal`/`-shm`) into a uniquely-named, timestamped
  dir under `~/Music/djay/Backups/autohotcue/` (never overwriting an earlier
  one) before its first write; a folder run takes one backup for the batch.
- **Overwrite protection** — tracks with existing cues need `--force`.
- **Self-checking writes** — every serialized blob is re-parsed and
  re-serialized and must match before it touches the database.
- **iCloud caveat:** djay syncs the library to iCloud through its own code.
  A direct DB write sets `userChangedCloudKeys` but bypasses djay's sync
  machinery, so externally-written cues may not propagate to other devices
  and could, in principle, be overwritten by a later iCloud pull. Safest on
  a single machine, or with djay's iCloud library sync off.

## Project layout

```
src/autohotcue/
    tsaf.py      # TSAF binary parser + serializer (byte-exact round-trip)
    djaydb.py    # MediaLibrary.db reader/writer, backup, cue record builder
    backends.py  # beat_this + librosa structure segmentation
    cuepolicy.py # pure cue-placement policy
    analysis.py  # decode, analyze() dispatcher, legacy path
    bench.py     # ground-truth eval harness
    viz.py       # waveform + segment spans + cue-map PNG
    cli.py       # propose / viz / apply / verify / bench
tests/
    test_tsaf.py # round-trips the entire real library
    test_e2e.py  # real-track integration tests
```

## Status

Built and validated against a real library on macOS (djay Pro, Catalyst
build). The cue record for *Adam Port, Stryv, Keinemusik, Orso, Malachiii —
Move (Extended Mix).opus* was written and confirmed intact across a djay
quit/relaunch cycle, proving djay parses and keeps the externally-written
record.
