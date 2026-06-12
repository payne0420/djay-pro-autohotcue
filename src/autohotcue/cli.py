"""autohotcue CLI — automatic hot cues for djay Pro's library.

Analyzes audio structure directly (works with any ffmpeg-decodable format,
including .opus / .ogg that rekordbox and most tools reject) and writes an
8-cue layout straight into djay's MediaLibrary.db.

Commands:
    propose  PATH          analyze and print proposed cues (no writes)
    viz      PATH OUT.png  render a waveform + cue map
    apply    PATH          write cues into djay's library (djay must be quit)
    verify   PATH          read back the cues djay has stored
    bench    TRUTH.json    score engines against hand-labeled ground truth

PATH for propose/apply/verify may be a single audio file or a directory,
which is scanned recursively. A directory `apply` takes one backup for the
whole run and reports per-track skips/failures instead of aborting on them.
`--jobs N` analyzes N tracks in parallel (worker processes); the database is
only ever touched from the main thread, so all write-path guards still hold.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sqlite3
import subprocess
from pathlib import Path

from autohotcue import analysis, backends, djaydb

AUDIO_EXTS = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3",
              ".oga", ".ogg", ".opus", ".wav", ".wma"}


class _Skip(Exception):
    """Per-track condition: skip this track in a batch, exit for a single file."""


def _expand_paths(path: str) -> list[str]:
    """A file is used as-is; a directory is walked recursively for audio files.

    Hidden files and directories (dot-prefixed, e.g. macOS ``._`` AppleDouble
    sidecars) are skipped.
    """
    p = Path(path)
    if not p.is_dir():
        return [path]
    files = sorted(
        str(f) for f in p.rglob("*")
        if f.is_file()
        and f.suffix.lower() in AUDIO_EXTS
        and not any(part.startswith(".") for part in f.relative_to(p).parts)
    )
    if not files:
        raise SystemExit(f"no audio files found under {p}")
    return files


def _resolve_jobs(n: int) -> int:
    return (os.cpu_count() or 1) if n == 0 else max(1, n)


def _pool_initargs(jobs: int) -> tuple:
    return (jobs,)


def _bpm_for(db: djaydb.DjayDB, key: str, fallback: float | None) -> float:
    """Pull djay's analyzed BPM for the track if present."""
    from autohotcue import tsaf

    doc = db.get("mediaItemAnalyzedData", key)
    if doc is not None:
        bpm = doc.root.get("bpm")
        if isinstance(bpm, tsaf.F32):
            return bpm.value
    if fallback:
        return fallback
    raise _Skip("no BPM in djay analysis and --bpm not given; analyze in djay first")


def _djay_running() -> bool:
    out = subprocess.run(["pgrep", "-x", "djay Pro"], capture_output=True, text=True)
    return out.returncode == 0


def _print_proposal(track: analysis.TrackAnalysis, prop: analysis.CueProposal) -> None:
    print(
        f"{track.engine}: {track.bpm:.1f} BPM, anchor {track.first_beat_s:.3f}s, "
        f"{track.duration_s:.1f}s"
    )
    for pad in "ABCDEFGH":
        t = prop.positions.get(pad)
        if t is not None:
            m, s = divmod(t, 60)
            print(f"  {pad} {djaydb.CUE_LABELS[ord(pad)-65]:16s} {int(m)}:{s:05.2f} ({t:.3f}s)")
    for note in prop.notes:
        print("  -", note)


def cmd_propose(args):
    paths = _expand_paths(args.path)
    batch = Path(args.path).is_dir()
    jobs = analysis.effective_parallel_jobs(args.engine, _resolve_jobs(args.jobs))
    failed = 0
    if batch and jobs > 1 and len(paths) > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=backends.init_worker,
            initargs=_pool_initargs(jobs),
        ) as ex:
            futs = {
                ex.submit(
                    analysis.analyze,
                    p,
                    args.bpm,
                    args.engine,
                    None,
                    jobs,
                ): p
                for p in paths
            }
            for fut in concurrent.futures.as_completed(futs):
                print(f"\n{Path(futs[fut]).name}")
                try:
                    track, prop = fut.result()
                except Exception as e:
                    print(f"  FAILED: {e}")
                    failed += 1
                    continue
                _print_proposal(track, prop)
    else:
        if jobs == 1:
            backends.init_worker(1)
        for path in paths:
            if batch:
                print(f"\n{Path(path).name}", flush=True)
            try:
                track, prop = analysis.analyze(
                    path,
                    known_bpm=args.bpm,
                    engine=args.engine,
                    jobs=1,
                )
            except Exception as e:
                if not batch:
                    raise
                print(f"  FAILED: {e}")
                failed += 1
                continue
            _print_proposal(track, prop)
    if batch:
        print(f"\n{len(paths) - failed} analyzed, {failed} failed ({len(paths)} files)")


def cmd_viz(args):
    from autohotcue import viz

    backends.init_worker(1)
    track, prop = analysis.analyze(
        args.path,
        known_bpm=args.bpm,
        engine=args.engine,
        jobs=1,
    )
    viz.render(args.path, track, prop, args.out, title=Path(args.path).stem)
    print("wrote", args.out)


def _precheck_one(db: djaydb.DjayDB, path: str, args):
    """Cheap DB-side checks before the (slow) analysis: locate the track and
    make sure its record is safe to edit. Returns (key, existing_doc, bpm)."""
    from autohotcue import tsaf

    try:
        key = db.find_track_by_path(path)
    except ValueError as e:
        raise _Skip(str(e)) from None
    if key is None:
        raise _Skip("track not found in djay library — import it into djay first "
                    "(add to My Collection)")

    # Read the existing record (if any) as raw bytes so we can guarantee we can
    # reproduce it byte-for-byte before mutating it — never corrupt other fields.
    raw = db.get_raw("mediaItemUserData", key)
    existing = None
    if raw is not None:
        existing = tsaf.parse(raw)
        if existing.root.get("cuePoints") and not args.force:
            raise _Skip("track already has cue points; pass --force to overwrite")
        try:
            faithful = tsaf.serialize(existing) == raw
        except Exception:
            faithful = False
        if not faithful:
            raise _Skip(
                "refusing to edit: this record uses a TSAF structure autohotcue "
                "cannot reproduce byte-exact, so editing it could corrupt other "
                "fields. Please report this track."
            )

    return key, existing, _bpm_for(db, key, args.bpm)


def _write_one(db: djaydb.DjayDB, key: str, existing, prop, ensure_backup) -> int:
    """Build the cue record from an analysis result and write it. Main thread
    only — this is the sole place apply touches the database after pre-checks."""
    from autohotcue import tsaf

    cues = []
    for i, pad in enumerate("ABCDEFGH"):
        t = prop.positions.get(pad)
        if t is not None:
            cues.append({"time": t, "number": i, "comment": djaydb.CUE_LABELS[i]})
    if not cues:
        raise _Skip("analysis produced no cues")

    # Re-check right before touching the DB: djay must not have started during
    # the (possibly slow) audio analysis.
    if _djay_running():
        raise SystemExit("djay Pro started during analysis — aborting before any write")

    ensure_backup()

    if existing is not None:
        doc = existing
        doc.root.set("cuePoints", tsaf.Arr(tsaf.TAG_ARRAY_A, djaydb.build_cue_objects(cues)))
        djaydb.ensure_cloud_key(doc.root, "cuePoints")
    else:
        tid = db.get("mediaItemTitleIDs", key)
        if tid is None:
            raise _Skip("no titleID record for this track; cannot build cue record")
        doc = djaydb.build_user_data(key, tid.root, cues)

    db.put("mediaItemUserData", key, doc)
    return len(cues)


def _apply_parallel(db, paths, args, ensure_backup, jobs):
    """Pre-check every track first (main thread), fan the analyses out to
    worker processes, then write each result back on the main thread as it
    completes. Worker processes never see the database."""
    total = len(paths)
    written = skipped = failed = 0
    todo = []
    for i, path in enumerate(paths, 1):
        try:
            key, existing, bpm = _precheck_one(db, path, args)
        except _Skip as e:
            print(f"[{i}/{total}] {Path(path).name}\n  skip: {e}")
            skipped += 1
            continue
        todo.append((i, path, key, existing, bpm))

    if todo:
        print(f"analyzing {len(todo)} tracks with {jobs} workers", flush=True)
        ex = concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=backends.init_worker,
            initargs=_pool_initargs(jobs),
        )
        try:
            futs = {
                ex.submit(
                    analysis.analyze,
                    path,
                    bpm,
                    args.engine,
                    None,
                    jobs,
                ): (i, path, key, existing)
                for i, path, key, existing, bpm in todo
            }
            for fut in concurrent.futures.as_completed(futs):
                i, path, key, existing = futs[fut]
                print(f"[{i}/{total}] {Path(path).name}", flush=True)
                try:
                    track, prop = fut.result()
                    n = _write_one(db, key, existing, prop, ensure_backup)
                except _Skip as e:
                    print(f"  skip: {e}")
                    skipped += 1
                except Exception as e:
                    print(f"  FAILED: {e}")
                    failed += 1
                else:
                    print(f"  wrote {n} cues")
                    written += 1
        finally:
            # On abort (djay started, Ctrl-C) drop queued analyses immediately;
            # nothing else gets written either way.
            ex.shutdown(wait=False, cancel_futures=True)
    return written, skipped, failed


def cmd_apply(args):
    if _djay_running():
        raise SystemExit("djay Pro is running — quit it first (it must not hold the DB open)")
    paths = _expand_paths(args.path)
    batch = Path(args.path).is_dir()
    jobs = analysis.effective_parallel_jobs(args.engine, _resolve_jobs(args.jobs))
    db = djaydb.DjayDB(args.library) if args.library else djaydb.DjayDB()

    backup_dir = args.backup_dir or (Path.home() / "Music/djay/Backups/autohotcue")
    backed_up = False

    def ensure_backup():
        # One backup per run, taken just before the first write, so a run where
        # every track is skipped leaves no backup behind.
        nonlocal backed_up
        if not backed_up:
            print("backup:", db.backup(backup_dir))
            backed_up = True

    total = len(paths)
    written = skipped = failed = 0
    try:
        if batch and jobs > 1 and total > 1:
            written, skipped, failed = _apply_parallel(db, paths, args, ensure_backup, jobs)
        else:
            if jobs == 1:
                backends.init_worker(1)
            for i, path in enumerate(paths, 1):
                name = Path(path).name
                if batch:
                    print(f"[{i}/{total}] {name}", flush=True)
                try:
                    key, existing, bpm = _precheck_one(db, path, args)
                    track, prop = analysis.analyze(
                        path,
                        known_bpm=bpm,
                        engine=args.engine,
                        jobs=1,
                    )
                    n = _write_one(db, key, existing, prop, ensure_backup)
                except _Skip as e:
                    if not batch:
                        raise SystemExit(str(e)) from None
                    print(f"  skip: {e}")
                    skipped += 1
                    continue
                except Exception as e:
                    if not batch:
                        raise
                    print(f"  FAILED: {e}")
                    failed += 1
                    continue
                print(f"  wrote {n} cues" if batch else f"wrote {n} cues to {name}")
                written += 1
        if batch:
            print(f"\n{written} written, {skipped} skipped, {failed} failed ({total} files)")
    finally:
        if backed_up:
            try:
                db.checkpoint()
            except sqlite3.OperationalError:
                pass  # djay may have started and grabbed the DB; writes are committed
        db.close()


def cmd_verify(args):
    from autohotcue import tsaf

    db = djaydb.DjayDB(args.library) if args.library else djaydb.DjayDB()
    paths = _expand_paths(args.path)
    batch = Path(args.path).is_dir()
    for path in paths:
        if batch:
            print(Path(path).name)
        try:
            key = db.find_track_by_path(path)
        except ValueError as e:
            if not batch:
                raise SystemExit(str(e)) from None
            print(f"  {e}")
            continue
        if key is None:
            if not batch:
                raise SystemExit("track not found in djay library")
            print("  not in djay library")
            continue
        doc = db.get("mediaItemUserData", key)
        if doc is None or not doc.root.get("cuePoints"):
            print("  no cues stored" if batch else "no cues stored for this track")
            continue
        for cp in doc.root.get("cuePoints").items:
            num = cp.get("number")
            n = num.value if isinstance(num, tsaf.Int) else (0 if num.tag == 0x2D else 1)
            print(f"  {chr(65+n)} {cp.get('comment') or '':16s} {cp.get('time').value:.3f}s")
    db.close()


_ENGINE_CHOICES = ("ml", "ml-librosa", "ml-allin1", "ml-songformer", "legacy")


def _add_engine_arg(sp):
    sp.add_argument(
        "--engine",
        choices=_ENGINE_CHOICES,
        default="ml",
        help="analysis engine: ml/ml-librosa (librosa structure), "
        "ml-allin1 (all-in-one-mlx structure), ml-songformer (SongFormer structure, CPU), "
        "legacy (default: ml)",
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="autohotcue", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    jobs_help = "parallel analysis workers for a folder (0 = one per CPU core)"

    sp = sub.add_parser("propose")
    sp.add_argument("path", help="audio file, or directory to scan recursively")
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--library", default=None)
    sp.add_argument("-j", "--jobs", type=int, default=1, help=jobs_help)
    _add_engine_arg(sp)

    sp = sub.add_parser("verify")
    sp.add_argument("path", help="audio file, or directory to scan recursively")
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--library", default=None)

    sp = sub.add_parser("viz")
    sp.add_argument("path", help="audio file")
    sp.add_argument("out")
    sp.add_argument("--bpm", type=float, default=None)
    _add_engine_arg(sp)

    sp = sub.add_parser("apply")
    sp.add_argument("path", help="audio file, or directory to scan recursively")
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--library", default=None)
    sp.add_argument("--backup-dir", default=None)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("-j", "--jobs", type=int, default=1, help=jobs_help)
    _add_engine_arg(sp)

    sp = sub.add_parser("bench")
    sp.add_argument("truth_json", help="ground-truth JSON file")
    sp.add_argument("--engines", default="ml,legacy", help="comma-separated engines to compare")
    sp.add_argument("--library", default=None, help="djay library for BPM yardstick lookup")
    sp.add_argument("--bpm", type=float, default=None, help="fallback BPM when not in library")
    sp.add_argument("-j", "--jobs", type=int, default=1, help=jobs_help)

    args = p.parse_args(argv)
    handlers = {
        "propose": cmd_propose,
        "viz": cmd_viz,
        "apply": cmd_apply,
        "verify": cmd_verify,
        "bench": lambda a: __import__("autohotcue.bench", fromlist=["cmd_bench"]).cmd_bench(a),
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
