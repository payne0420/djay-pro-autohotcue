"""Read/write access to djay Pro's MediaLibrary.db (YapDatabase on SQLite).

All object payloads are TSAF blobs (see tsaf.py). Records are keyed by a
content-derived track hash shared across collections (mediaItemTitleIDs,
mediaItemAnalyzedData, mediaItemUserData, localMediaItemLocations).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

from autohotcue import tsaf

DEFAULT_LIBRARY = Path.home() / "Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db"

# Cue slot labels from the djcues cue system (pads A-H = numbers 0-7)
CUE_LABELS = [
    "First Beat", "Loop In", "Vocal / Buildup", "Drop",
    "Breakdown", "Special", "Outro", "Loop Out",
]


def cue_number_value(n: int):
    """djay's integer encoding convention for cue slot numbers."""
    if n == 0:
        return tsaf.Marker(tsaf.TAG_M2D)
    if n == 1:
        return tsaf.Marker(tsaf.TAG_M2E)
    return tsaf.Int(tsaf.TAG_INT8, n)


def _file_url_to_path(url: str) -> str | None:
    """Decode a ``file://`` URL to an absolute filesystem path."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def _path_variants(path: str) -> set[str]:
    """Comparable forms of a path: absolute and symlink-resolved, both NFC.

    macOS file systems and djay's stored URLs disagree on Unicode normalization
    for accented names (NFC vs NFD), so all comparisons go through NFC.
    """
    variants = {os.path.abspath(path)}
    try:
        variants.add(os.path.realpath(path))
    except OSError:
        pass
    return {unicodedata.normalize("NFC", v) for v in variants}


def build_cue_objects(cues: list[dict]) -> list[tsaf.Obj]:
    """Build ``ADCCuePoint`` objects (sorted by slot) from cue dicts.

    Each dict: {time: float, number: int, comment: str|None}. endTime is -1
    (non-loop). colorIndex is omitted — djay assigns a default per slot, and
    minimal real records (time/endTime/number only) are valid.
    """
    objs = []
    for c in sorted(cues, key=lambda c: c["number"]):
        cue = tsaf.Obj("ADCCuePoint")
        cue.fields.append(("time", tsaf.F32.of(c["time"])))
        cue.fields.append(("endTime", tsaf.F32.of(-1.0)))
        if c.get("comment"):
            cue.fields.append(("comment", c["comment"]))
        cue.fields.append(("number", cue_number_value(c["number"])))
        objs.append(cue)
    return objs


def ensure_cloud_key(root: tsaf.Obj, key: str):
    """Make sure djay's userChangedCloudKeys array contains ``key``."""
    arr = root.get("userChangedCloudKeys")
    if isinstance(arr, tsaf.Arr):
        if key not in arr.items:
            arr.items.append(key)
    else:
        root.set("userChangedCloudKeys", tsaf.Arr(tsaf.TAG_ARRAY_B, [key]))


class DjayDB:
    def __init__(self, path: Path | str = DEFAULT_LIBRARY):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self._path_index: dict[str, set[str]] | None = None

    def close(self):
        self.conn.close()

    def backup(self, dest_dir: Path | str) -> Path:
        """Copy db + sidecar files to a unique timestamped backup directory.

        Never reuses an existing directory: a second run within the same second
        gets a ``-1``/``-2`` suffix so an earlier restore point is preserved.
        Checkpoints WAL into the main db first so the copy is self-consistent.
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except sqlite3.OperationalError:
            pass  # busy/locked — fall back to copying db+wal+shm together
        base = Path(dest_dir) / time.strftime("%Y%m%d-%H%M%S")
        dest, n = base, 1
        while dest.exists():
            dest = Path(f"{base}-{n}")
            n += 1
        dest.mkdir(parents=True)
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self.path) + suffix)
            if src.exists():
                shutil.copy2(src, dest / src.name)
        return dest

    def get(self, collection: str, key: str) -> tsaf.Document | None:
        raw = self.get_raw(collection, key)
        return tsaf.parse(raw) if raw is not None else None

    def get_raw(self, collection: str, key: str) -> bytes | None:
        """Return the stored blob bytes verbatim (for byte-exact comparisons)."""
        row = self.conn.execute(
            "SELECT data FROM database2 WHERE collection=? AND key=?",
            (collection, key),
        ).fetchone()
        return bytes(row[0]) if row is not None and row[0] is not None else None

    def put(self, collection: str, key: str, doc: tsaf.Document):
        """Insert or replace a record. Round-trip-verifies the blob first."""
        blob = tsaf.serialize(doc)
        reparsed = tsaf.serialize(tsaf.parse(blob))
        if reparsed != blob:
            raise ValueError("serialized blob failed round-trip self-check")
        with self.conn:
            self.conn.execute(
                "INSERT INTO database2 (collection, key, data, metadata) "
                "VALUES (?, ?, ?, NULL) "
                "ON CONFLICT(collection, key) DO UPDATE SET data=excluded.data",
                (collection, key, blob),
            )

    def checkpoint(self):
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def _build_path_index(self) -> dict[str, set[str]]:
        """Map every stored source path (normalized variants) to its track keys."""
        idx: dict[str, set[str]] = {}
        rows = self.conn.execute(
            "SELECT key, data FROM database2 WHERE collection='localMediaItemLocations'"
        ).fetchall()
        for key, data in rows:
            try:
                doc = tsaf.parse(bytes(data))
            except Exception:
                continue
            uris = doc.root.get("sourceURIs")
            if not isinstance(uris, tsaf.Arr):
                continue
            for u in uris.items:
                url = u.value if isinstance(u, tsaf.Url) else (u if isinstance(u, str) else None)
                path = _file_url_to_path(url) if url else None
                if not path:
                    continue
                for variant in _path_variants(path):
                    idx.setdefault(variant, set()).add(key)
        return idx

    def find_track_by_path(self, file_path: str) -> str | None:
        """Return the unique track key whose stored source URL is this exact file.

        Builds a path -> keys index over all location records on first use
        (per connection) and matches the absolute/resolved, NFC-normalized
        path exactly. Raises ValueError if more than one distinct track
        matches — better to refuse than to write cues to the wrong track.
        Returns None if nothing matches.
        """
        if self._path_index is None:
            self._path_index = self._build_path_index()
        matches: set[str] = set()
        for variant in _path_variants(file_path):
            matches |= self._path_index.get(variant, set())
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous: {len(matches)} library entries match {file_path}; refusing to guess"
            )
        return next(iter(matches)) if matches else None


def build_user_data(key: str, title_id: tsaf.Obj, cues: list[dict]) -> tsaf.Document:
    """Build a fresh ADCMediaItemUserData document.

    cues: list of {time: float, number: int, comment: str|None}, endTime -1.
    Field order and value encodings mirror records djay itself writes.
    """
    cue_objs = build_cue_objects(cues)

    root = tsaf.Obj("ADCMediaItemUserData")
    root.fields.append(("uuid", key))
    root.fields.append((
        "userChangedCloudKeys",
        tsaf.Arr(tsaf.TAG_ARRAY_B, ["cuePoints", "titleIDs", "playCount"]),
    ))
    root.fields.append(("titleIDs", tsaf.Arr(tsaf.TAG_ARRAY_B, [title_id])))
    root.fields.append(("cuePoints", tsaf.Arr(tsaf.TAG_ARRAY_A, cue_objs)))
    root.fields.append(("playCount", tsaf.Marker(tsaf.TAG_M2D)))
    return tsaf.Document((3, 3), root)
