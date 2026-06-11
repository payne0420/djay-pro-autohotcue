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
2. **Lock a beatgrid** to djay's analyzed BPM (phase + first downbeat found
   from the onset envelope and low-band kick energy).
3. **Find structure** from per-bar band-split energy: the drop (bass returns
   after a buildup), the buildup (bass cuts out), the breakdown (sustained
   bass dropout), the second drop, and the outro. Everything snaps to phrase
   boundaries on the grid.
4. **Write cues** into djay's `mediaItemUserData` record as `ADCCuePoint`
   objects, serialized in djay's own **TSAF** binary format.

### The cue layout

| Pad | Cue            | Anchored on |
|-----|----------------|-------------|
| A   | First Beat     | grid anchor |
| B   | Loop In        | grid anchor |
| C   | Vocal / Buildup| bass cuts out before the drop |
| D   | Drop           | bass returns / first sustained energy peak |
| E   | Breakdown      | sustained bass dropout after the drop |
| F   | Special        | energy recovery / second drop |
| G   | Outro          | energy tail begins |
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

```bash
uv sync

# Analyze and print proposed cues (no writes, no djay needed)
uv run autohotcue propose "/path/to/track.opus"

# Render a waveform + cue map PNG
uv run autohotcue viz "/path/to/track.opus" cuemap.png

# Write cues into djay's library (QUIT djay first)
uv run autohotcue apply "/path/to/track.opus"

# Read back what djay has stored for a track
uv run autohotcue verify "/path/to/track.opus"

# Point propose/apply/verify at a folder to process every audio file in it
# (recursive). A folder apply takes ONE backup for the whole run and skips
# problem tracks (not in djay, already has cues, ...) with a summary.
# -j N analyzes N tracks in parallel (-j 0 = one worker per CPU core);
# database writes always stay serialized on the main thread.
uv run autohotcue apply "/path/to/Afro House/" -j 4
```

The track must already be in djay's **My Collection** (so the library has a
record to attach cues to). `apply` reuses djay's analyzed BPM automatically;
pass `--bpm` to override.

## Safety

- **djay must be quit** before `apply` — it refuses if `djay Pro` is running,
  and **re-checks immediately before the write** in case djay started during
  the (slow) audio analysis.
- **Exact track matching** — the track is matched by its full `file://` source
  URL, and an ambiguous match (more than one library entry) is refused rather
  than guessed, so cues never land on the wrong track.
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
    analysis.py  # ffmpeg decode, beatgrid lock, structure detection
    viz.py       # waveform + cue-map PNG
    cli.py       # propose / viz / apply / verify
tests/
    test_tsaf.py # round-trips the entire real library
```

## Status

Built and validated against a real library on macOS (djay Pro, Catalyst
build). The cue record for *Adam Port, Stryv, Keinemusik, Orso, Malachiii —
Move (Extended Mix).opus* was written and confirmed intact across a djay
quit/relaunch cycle, proving djay parses and keeps the externally-written
record.
