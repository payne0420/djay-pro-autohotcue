# AGENTS.md

Guidance for coding agents working on **autohotcue**. Read this before changing
code — this project writes directly into a DJ app's real, irreplaceable music
library, so some rules below are load-bearing, not stylistic.

## Project overview

autohotcue places "hot cue" markers on tracks automatically and writes them
into **djay Pro**'s library database. It analyzes audio structure directly
(any `ffmpeg`-decodable format, including `.opus`/`.ogg`), so it does not depend
on another DJ app's analysis.

Pure-Python, single package, no service. Default pipeline (`ml-bass`): decode
audio → **beat_this** beat/downbeat tracking → `gridlock` straight-grid fit →
**kick-band (30–150 Hz) bass-in/out/return detection + 16/32-bar phrase snapping**
(`bassline`) → write an 8-cue layout into djay's `MediaLibrary.db`. The structure
engines segment by song form instead: `--engine ml`/`ml-librosa` use librosa
Laplacian segmentation + the pure cue policy, `--engine ml-allin1` swaps structure
for **all-in-one-mlx** (Apple Silicon; `uv sync --extra allin1`),
`--engine ml-songformer` swaps structure for **SongFormer** (CPU by default;
`uv sync --extra songformer`). A legacy band-energy engine remains behind
`--engine legacy`.

```
src/autohotcue/
    tsaf.py      # parser + serializer for djay's undocumented "TSAF" blob format
    djaydb.py    # MediaLibrary.db (SQLite/YapDatabase) reader/writer, backups, cue builder
    backends.py  # beat_this + librosa / allin1 / songformer structure backends
    _allin1.py   # optional all-in-one-mlx structure (ml-allin1 engine)
    _songformer.py  # optional SongFormer structure (ml-songformer engine)
    gridlock.py  # pure straight-grid fit + cue snapping (beat/downbeat -> lattice)
    bassline.py  # pure kick-band bass events + phrase snapping (ml-bass policy)
    cuepolicy.py # pure structure-based cue-placement policy (no audio, no I/O)
    analysis.py  # decode, analyze() dispatcher, legacy grid path
    bench.py     # ground-truth eval harness (hit-rate, MAE, runtime)
    viz.py       # waveform + segment/bass spans + downbeat ticks + cue-map PNG
    cli.py       # propose / viz / apply / verify / bench
tests/
    test_tsaf.py     # byte-exact round-trip of the whole real library + edge cases
    test_gridlock.py # straight-grid fit, snapping, beatGridEdits encoding
    test_bassline.py # ml-bass kick-band events + phrase snapping + slot rules
    test_cuepolicy.py
    test_backends.py
    test_bench.py
    test_e2e.py      # real-track integration (skipped when paths missing)
```

Key tech: Python 3.13 (pinned; `requires-python >=3.11`), [uv](https://docs.astral.sh/uv/)
for env/deps, numpy + librosa + soundfile + matplotlib + **PyTorch** (via
`beat-this`), and the external `ffmpeg` binary (must be on `PATH`). First run
downloads beat_this model checkpoints (one-time network).

## Setup commands

- Install deps + create venv: `uv sync`
- Requires `ffmpeg` on `PATH` (`ffmpeg -version`); install via Homebrew: `brew install ffmpeg`
- Python is pinned in `.python-version` (3.13.3); `uv` provisions it.

## Development workflow

Run the CLI through uv (no manual venv activation needed):

```bash
uv run autohotcue propose "/path/to/track.opus"              # ml-bass engine (default)
uv run autohotcue propose "/path/to/track.opus" --engine ml         # librosa structure
uv run autohotcue propose "/path/to/track.opus" --engine ml-allin1  # all-in-one structure
uv run autohotcue propose "/path/to/track.opus" --engine ml-songformer  # SongFormer structure
uv run autohotcue propose "/path/to/track.opus" --nudge-beats 1     # shift grid anchor +1 beat
uv run autohotcue propose "/path/to/track.opus" --engine legacy
uv run autohotcue viz "/path/to/track.opus" map.png          # segments + downbeat ticks
uv run autohotcue verify "/path/to/track.opus"               # read back cues djay has stored
uv run autohotcue apply "/path/to/track.opus"                # WRITE cues into djay's DB
uv run autohotcue bench truth.json --engines ml,ml-allin1,legacy -j 4
```

`--engine {ml-bass,ml,ml-librosa,ml-allin1,ml-songformer,legacy}` (default `ml-bass`).
`ml-bass` (the default) uses kick-band bass events + 16/32-bar phrase snapping on the
grid-locked lattice; `ml` / `ml-librosa` use librosa Laplacian structure; `ml-allin1`
uses all-in-one-mlx (optional extra; see `docs/allin1-backend.md`); `ml-songformer` uses
SongFormer (optional extra; see `docs/songformer-backend.md`; CPU by default). djay's BPM
is cross-checked only under ml engines; placement uses beat_this downbeats. `apply` still
writes absolute seconds.

Use `--library <path>` to target a copy of the DB (do this when testing writes).

`propose`, `apply` and `verify` also accept a **folder** (scanned recursively
for audio files). A folder `apply` takes one backup for the whole run and
reports per-track skips/failures plus a summary instead of aborting.
`-j/--jobs N` fans the audio analysis out to N worker processes (0 = one per
core); the database is only ever touched from the main thread — keep it that
way if you change the parallel path.

### Parallelism rules

- `ProcessPoolExecutor` workers use `initializer=backends.init_worker` to cap
  torch threads per worker.
- Workers always run models on **cpu**; **mps** is used only when effective
  jobs == 1 (single-threaded analysis).
- One lazy model singleton per worker process (macOS spawn reloads each worker).
- Database writes stay on the main thread only.

## Testing instructions

- Run everything: `uv run pytest` (config pins `testpaths = ["tests"]`).
- Focus one test: `uv run pytest -q -k roundtrip`.
- The headline test (`test_roundtrip_entire_library`) parses **and re-serializes
  every blob** in the local djay library and asserts byte-for-byte equality. It
  **skips** automatically when no djay library is present (e.g. CI) — the
  synthetic tests still run. If you have djay installed, this test must pass.
- `tests/test_e2e.py` runs against real tracks when present on the machine.
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
- Every `apply` run takes a **unique, timestamped backup** of the DB
  (+ `-wal`/`-shm`) before its first write (one backup per folder batch).
- Track lookup matches the **exact** `file://` source URL and refuses ambiguous
  matches (phantom location records without a titleID lose to the real entry) —
  never write cues to the wrong track.

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
