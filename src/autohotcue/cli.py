"""autohotcue CLI — automatic hot cues for djay Pro's library.

Analyzes audio structure directly (works with any ffmpeg-decodable format,
including .opus / .ogg that rekordbox and most tools reject) and writes an
8-cue layout straight into djay's MediaLibrary.db.

Commands:
    propose  PATH          analyze and print proposed cues (no writes)
    viz      PATH OUT.png  render a waveform + cue map
    apply    PATH          write cues into djay's library (djay must be quit)
    verify   PATH          read back the cues djay has stored

PATH for propose/apply/verify may be a single audio file or a directory,
which is scanned recursively. A directory `apply` takes one backup for the
whole run and reports per-track skips/failures instead of aborting on them.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path

from autohotcue import analysis, djaydb

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


def cmd_propose(args):
    paths = _expand_paths(args.path)
    batch = Path(args.path).is_dir()
    failed = 0
    for path in paths:
        if batch:
            print(f"\n{Path(path).name}")
        try:
            grid, prop = analysis.propose_cues(path, known_bpm=args.bpm or 120.0)
        except Exception as e:
            if not batch:
                raise
            print(f"  FAILED: {e}")
            failed += 1
            continue
        print(f"grid: {grid.bpm:.1f} BPM, anchor {grid.first_beat_s:.3f}s, {grid.duration_s:.1f}s")
        for pad in "ABCDEFGH":
            t = prop.positions.get(pad)
            if t is not None:
                m, s = divmod(t, 60)
                print(f"  {pad} {djaydb.CUE_LABELS[ord(pad)-65]:16s} {int(m)}:{s:05.2f} ({t:.3f}s)")
        for note in prop.notes:
            print("  -", note)
    if batch:
        print(f"\n{len(paths) - failed} analyzed, {failed} failed ({len(paths)} files)")


def cmd_viz(args):
    from autohotcue import viz

    grid, prop = analysis.propose_cues(args.path, known_bpm=args.bpm or 120.0)
    viz.render(args.path, grid, prop, args.out, title=Path(args.path).stem)
    print("wrote", args.out)


def _apply_one(db: djaydb.DjayDB, path: str, args, ensure_backup) -> int:
    """Analyze one track and write its cues; returns the number of cues written.

    Raises _Skip for per-track conditions; SystemExit only for whole-run
    conditions (djay started while we were analyzing).
    """
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

    bpm = _bpm_for(db, key, args.bpm)
    grid, prop = analysis.propose_cues(path, known_bpm=bpm)
    cues = []
    for i, pad in enumerate("ABCDEFGH"):
        t = prop.positions.get(pad)
        if t is not None:
            cues.append({"time": t, "number": i, "comment": djaydb.CUE_LABELS[i]})
    if not cues:
        raise _Skip("analysis produced no cues")

    # Re-check right before touching the DB: djay must not have started during
    # the (possibly slow) audio analysis above.
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


def cmd_apply(args):
    if _djay_running():
        raise SystemExit("djay Pro is running — quit it first (it must not hold the DB open)")
    paths = _expand_paths(args.path)
    batch = Path(args.path).is_dir()
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

    written = skipped = failed = 0
    try:
        for i, path in enumerate(paths, 1):
            name = Path(path).name
            if batch:
                print(f"[{i}/{len(paths)}] {name}", flush=True)
            try:
                n = _apply_one(db, path, args, ensure_backup)
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
            print(f"\n{written} written, {skipped} skipped, {failed} failed ({len(paths)} files)")
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


def main(argv=None):
    p = argparse.ArgumentParser(prog="autohotcue", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("propose", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("path", help="audio file, or directory to scan recursively")
        sp.add_argument("--bpm", type=float, default=None)
        sp.add_argument("--library", default=None)

    sp = sub.add_parser("viz")
    sp.add_argument("path", help="audio file")
    sp.add_argument("out")
    sp.add_argument("--bpm", type=float, default=None)

    sp = sub.add_parser("apply")
    sp.add_argument("path", help="audio file, or directory to scan recursively")
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--library", default=None)
    sp.add_argument("--backup-dir", default=None)
    sp.add_argument("--force", action="store_true")

    args = p.parse_args(argv)
    {"propose": cmd_propose, "viz": cmd_viz, "apply": cmd_apply, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
