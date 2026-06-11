# AGENTS.md

Guidance for coding agents working on **autohotcue**. Read this before changing
code — this project writes directly into a DJ app's real, irreplaceable music
library, so some rules below are load-bearing, not stylistic.

## Project overview

autohotcue places "hot cue" markers on tracks automatically and writes them
into **djay Pro**'s library database. It analyzes audio structure directly
(any `ffmpeg`-decodable format, including `.opus`/`.ogg`), so it does not depend
on another DJ app's analysis.

Pure-Python, single package, no service. Pipeline: decode audio → lock a
beatgrid → detect structure (drop/breakdown/buildup/outro) → write an 8-cue
layout into djay's `MediaLibrary.db`.

```
src/autohotcue/
    tsaf.py      # parser + serializer for djay's undocumented "TSAF" blob format
    djaydb.py    # MediaLibrary.db (SQLite/YapDatabase) reader/writer, backups, cue builder
    analysis.py  # ffmpeg decode, beatgrid lock, band-energy structure detection
    viz.py       # waveform + cue-map PNG
    cli.py       # propose / viz / apply / verify
tests/
    test_tsaf.py # byte-exact round-trip of the whole real library + edge cases
```

Key tech: Python 3.13 (pinned; `requires-python >=3.11`), [uv](https://docs.astral.sh/uv/)
for env/deps, numpy + librosa + soundfile + matplotlib, and the external
`ffmpeg` binary (must be on `PATH`).

## Setup commands

- Install deps + create venv: `uv sync`
- Requires `ffmpeg` on `PATH` (`ffmpeg -version`); install via Homebrew: `brew install ffmpeg`
- Python is pinned in `.python-version` (3.13.3); `uv` provisions it.

## Development workflow

Run the CLI through uv (no manual venv activation needed):

```bash
uv run autohotcue propose "/path/to/track.opus"     # analyze + print cues, no writes
uv run autohotcue viz "/path/to/track.opus" map.png # waveform + cue-map PNG
uv run autohotcue verify "/path/to/track.opus"      # read back cues djay has stored
uv run autohotcue apply "/path/to/track.opus"       # WRITE cues into djay's DB
```

`apply` reuses djay's analyzed BPM automatically; override with `--bpm`.
Use `--library <path>` to target a copy of the DB (do this when testing writes).

## Testing instructions

- Run everything: `uv run pytest` (config pins `testpaths = ["tests"]`).
- Focus one test: `uv run pytest -q -k roundtrip`.
- The headline test (`test_roundtrip_entire_library`) parses **and re-serializes
  every blob** in the local djay library and asserts byte-for-byte equality. It
  **skips** automatically when no djay library is present (e.g. CI) — the
  synthetic tests still run. If you have djay installed, this test must pass.
- Add or update tests for any change to `tsaf.py` or `djaydb.py`.

## The non-negotiable safety invariant

`tsaf.serialize(tsaf.parse(blob)) == blob` must hold for **every** blob in a
real djay library. This byte-exact round-trip is the only thing that makes
writing to the live database safe. If you touch `tsaf.py`:

1. Run the full-library round-trip test before and after.
2. Never make the serializer "lossy but close" — a one-byte drift can corrupt
   the user's entire catalog.
3. The write path (`djaydb.put`, `cli.cmd_apply`) already refuses to edit any
   record it cannot reproduce byte-exact (the "faithful-edit guard"). Do not
   weaken or bypass that guard.

Other write-path rules already enforced — keep them:
- **djay must be quit** before `apply` (it holds the DB open); it is re-checked
  immediately before the write.
- Every `apply` takes a **unique, timestamped backup** of the DB (+ `-wal`/`-shm`)
  before writing.
- Track lookup matches the **exact** `file://` source URL and refuses ambiguous
  matches — never write cues to the wrong track.

## Code style

- Match the surrounding code: type hints, `from __future__ import annotations`,
  small focused functions, stdlib-first.
- Comments only state constraints the code can't show (e.g. the TSAF tag table,
  alignment rules). Don't narrate what the next line does.
- No formatter/linter is configured; follow existing formatting (4-space indent,
  ~100-col lines, double-quoted strings).
- Keep `tsaf.py` free of project-specific logic — it is a generic codec.

## Build and deployment

There is no build/deploy step — it's a local CLI. `uv build` produces a wheel if
needed. No CI is configured yet; if you add one, wire `uv sync` + `uv run pytest`
(the library round-trip test will skip in CI, which is expected).

## Pull request / commit guidelines

- Branch off and target `main`.
- Commits in this repo use a noreply identity (no personal email in history);
  preserve that convention.
- Before committing: `uv run pytest` must be green.
- Never commit `.venv/`, the user's `MediaLibrary.db`, or any backup of it.

## Gotchas

- `.opus`/`.ogg` are first-class here precisely because rekordbox/Serato reject
  them — don't assume a format is unsupported; if `ffmpeg` decodes it, it works.
- djay Pro is a Catalyst (iOS-on-Mac) app: its library UI rejects synthetic
  accessibility/keystroke input, so end-to-end GUI automation of "load a track
  and screenshot the cues" is not reliable. Verify writes at the DB level
  (`verify` command / the round-trip test) instead.
- djay syncs to iCloud via its own code path; externally-written cues set
  `userChangedCloudKeys` but bypass that machinery, so they may not propagate to
  other devices. Treat single-machine as the supported case.
- The TSAF container-count header counts objects, arrays/sets, NSData and NSDate
  (f64) but **not** URLs/strings/scalars — see the module docstring in `tsaf.py`.
