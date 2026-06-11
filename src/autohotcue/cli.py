"""autohotcue CLI — automatic hot cues for djay Pro's library.

Analyzes audio structure directly (works with any ffmpeg-decodable format,
including .opus / .ogg that rekordbox and most tools reject) and writes an
8-cue layout straight into djay's MediaLibrary.db.

Commands:
    propose  PATH         analyze a file and print proposed cues (no writes)
    viz      PATH OUT.png  render a waveform + cue map
    apply    PATH          write cues into djay's library (djay must be quit)
    verify   PATH          read back the cues djay has for a file
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from autohotcue import analysis, djaydb


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
    raise SystemExit("no BPM in djay analysis and --bpm not given; analyze in djay first")


def _djay_running() -> bool:
    out = subprocess.run(["pgrep", "-x", "djay Pro"], capture_output=True, text=True)
    return out.returncode == 0


def cmd_propose(args):
    grid, prop = analysis.propose_cues(args.path, known_bpm=args.bpm or 120.0)
    print(f"grid: {grid.bpm:.1f} BPM, anchor {grid.first_beat_s:.3f}s, {grid.duration_s:.1f}s")
    for pad in "ABCDEFGH":
        t = prop.positions.get(pad)
        if t is not None:
            m, s = divmod(t, 60)
            print(f"  {pad} {djaydb.CUE_LABELS[ord(pad)-65]:16s} {int(m)}:{s:05.2f} ({t:.3f}s)")
    for note in prop.notes:
        print("  -", note)


def cmd_viz(args):
    from autohotcue import viz

    grid, prop = analysis.propose_cues(args.path, known_bpm=args.bpm or 120.0)
    viz.render(args.path, grid, prop, args.out, title=Path(args.path).stem)
    print("wrote", args.out)


def cmd_apply(args):
    from autohotcue import tsaf

    if _djay_running():
        raise SystemExit("djay Pro is running — quit it first (it must not hold the DB open)")
    db = djaydb.DjayDB(args.library) if args.library else djaydb.DjayDB()
    try:
        key = db.find_track_by_path(args.path)
    except ValueError as e:
        raise SystemExit(str(e))
    if key is None:
        raise SystemExit(f"track not found in djay library: {Path(args.path).name}\n"
                         "Import it into djay first (add to My Collection).")

    # Read the existing record (if any) as raw bytes so we can guarantee we can
    # reproduce it byte-for-byte before mutating it — never corrupt other fields.
    raw = db.get_raw("mediaItemUserData", key)
    existing = None
    if raw is not None:
        existing = tsaf.parse(raw)
        if existing.root.get("cuePoints") and not args.force:
            raise SystemExit("track already has cue points; pass --force to overwrite")
        try:
            faithful = tsaf.serialize(existing) == raw
        except Exception:
            faithful = False
        if not faithful:
            raise SystemExit(
                "refusing to edit: this record uses a TSAF structure autohotcue "
                "cannot reproduce byte-exact, so editing it could corrupt other "
                "fields. Please report this track."
            )

    bpm = _bpm_for(db, key, args.bpm)
    grid, prop = analysis.propose_cues(args.path, known_bpm=bpm)
    cues = []
    for i, pad in enumerate("ABCDEFGH"):
        t = prop.positions.get(pad)
        if t is not None:
            cues.append({"time": t, "number": i, "comment": djaydb.CUE_LABELS[i]})
    if not cues:
        raise SystemExit("analysis produced no cues; aborting")

    # Re-check right before touching the DB: djay must not have started during
    # the (possibly slow) audio analysis above.
    if _djay_running():
        raise SystemExit("djay Pro started during analysis — aborting before any write")

    backup = db.backup(args.backup_dir or (Path.home() / "Music/djay/Backups/autohotcue"))
    print("backup:", backup)

    if existing is not None:
        doc = existing
        doc.root.set("cuePoints", tsaf.Arr(tsaf.TAG_ARRAY_A, djaydb.build_cue_objects(cues)))
        djaydb.ensure_cloud_key(doc.root, "cuePoints")
    else:
        tid = db.get("mediaItemTitleIDs", key)
        if tid is None:
            raise SystemExit("no titleID record for this track; cannot build cue record")
        doc = djaydb.build_user_data(key, tid.root, cues)

    db.put("mediaItemUserData", key, doc)
    db.checkpoint()
    print(f"wrote {len(cues)} cues to {Path(args.path).name}")
    db.close()


def cmd_verify(args):
    db = djaydb.DjayDB(args.library) if args.library else djaydb.DjayDB()
    key = db.find_track_by_path(args.path)
    if key is None:
        raise SystemExit("track not found in djay library")
    doc = db.get("mediaItemUserData", key)
    if doc is None or not doc.root.get("cuePoints"):
        print("no cues stored for this track")
        return
    from autohotcue import tsaf

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
        sp.add_argument("path")
        sp.add_argument("--bpm", type=float, default=None)
        sp.add_argument("--library", default=None)

    sp = sub.add_parser("viz")
    sp.add_argument("path")
    sp.add_argument("out")
    sp.add_argument("--bpm", type=float, default=None)

    sp = sub.add_parser("apply")
    sp.add_argument("path")
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--library", default=None)
    sp.add_argument("--backup-dir", default=None)
    sp.add_argument("--force", action="store_true")

    args = p.parse_args(argv)
    {"propose": cmd_propose, "viz": cmd_viz, "apply": cmd_apply, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
