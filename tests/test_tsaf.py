"""TSAF parser/serializer round-trip tests against a real djay library.

The serializer MUST reproduce every blob byte-for-byte — anything less risks
corrupting the user's library on write. If a djay DB is present, validate the
whole thing; otherwise fall back to synthetic structures.
"""
import sqlite3
from pathlib import Path

import pytest

from autohotcue import tsaf, djaydb

LIB = Path.home() / "Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db"


def _blobs():
    if not LIB.exists():
        return []
    conn = sqlite3.connect(f"file:{LIB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT collection, key, data FROM database2 WHERE data IS NOT NULL"
    ).fetchall()
    conn.close()
    return [(c, k, bytes(d)) for c, k, d in rows if bytes(d)[:4] == b"TSAF"]


@pytest.mark.skipif(not LIB.exists(), reason="no djay library on this machine")
def test_roundtrip_entire_library():
    blobs = _blobs()
    assert blobs, "expected at least one TSAF blob"
    failures = []
    for coll, key, buf in blobs:
        doc = tsaf.parse(buf)
        if tsaf.serialize(doc) != buf:
            failures.append(f"{coll}/{key}")
    assert not failures, f"{len(failures)} blobs failed round-trip: {failures[:5]}"


def test_build_user_data_is_wellformed():
    tid = tsaf.Obj("ADCMediaItemTitleID")
    tid.fields = [("uuid", "deadbeef"), ("title", "T"), ("artist", "A"),
                  ("duration", tsaf.F32.of(200.0))]
    cues = [{"time": 1.5, "number": 0, "comment": "First Beat"},
            {"time": 60.0, "number": 3, "comment": "Drop"}]
    doc = djaydb.build_user_data("deadbeef", tid, cues)
    blob = tsaf.serialize(doc)
    # serialize -> parse -> serialize must be stable
    assert tsaf.serialize(tsaf.parse(blob)) == blob
    cps = tsaf.parse(blob).root.get("cuePoints")
    assert len(cps.items) == 2
    assert cps.items[0].get("time").value == pytest.approx(1.5)
    assert cps.items[0].get("endTime").value == -1.0


def test_int16_roundtrip_both_parities():
    """Tag 0x10 (u16, 2-aligned) — djay writes it for ``year`` in mediaItems.

    Vary the preceding string length so the u16 payload starts once on an odd
    offset (forcing a pad byte) and once on an even one; both must be stable.
    """
    for name in ("a", "ab"):
        obj = tsaf.Obj("X")
        obj.fields = [(name, tsaf.Int(tsaf.TAG_INT16, 2026))]
        blob = tsaf.serialize(tsaf.Document((3, 3), obj))
        rt = tsaf.parse(blob)
        assert tsaf.serialize(rt) == blob
        got = rt.root.get(name)
        assert got.tag == tsaf.TAG_INT16 and got.value == 2026


def test_cue_number_encoding():
    assert isinstance(djaydb.cue_number_value(0), tsaf.Marker)
    assert isinstance(djaydb.cue_number_value(1), tsaf.Marker)
    assert djaydb.cue_number_value(5).value == 5


def test_anonymous_field_is_stable():
    """Objects with an anonymous (unnamed) field must serialize stably."""
    obj = tsaf.Obj("X")
    obj.fields = [("named", tsaf.Int(tsaf.TAG_INT8, 7)), (None, tsaf.F32.of(2.0))]
    doc = tsaf.Document((3, 3), obj)
    once = tsaf.serialize(doc)
    assert tsaf.serialize(tsaf.parse(once)) == once


def test_url_and_data_roundtrip():
    obj = tsaf.Obj("L")
    obj.fields = [
        ("u", tsaf.Url("file:///tmp/a%20b.opus")),
        ("d", tsaf.Data(b"\x01\x02\x03")),  # length 3 -> exercises post-padding
        ("flag", True),
    ]
    doc = tsaf.Document((3, 3), obj)
    blob = tsaf.serialize(doc)
    rt = tsaf.parse(blob)
    assert tsaf.serialize(rt) == blob
    assert isinstance(rt.root.get("u"), tsaf.Url)
    assert rt.root.get("d").value == b"\x01\x02\x03"


def test_repeat_of_high_index_string_fails_loudly():
    """A string first interned past index 255 cannot be back-referenced with a
    u8 index. Repeating it must raise loudly rather than emit a wrong byte."""
    obj = tsaf.Obj("Many")
    # 300 distinct strings (all inlined fine), then repeat one interned > 255.
    obj.fields = [(None, f"s{i}") for i in range(300)] + [(None, "s260")]
    with pytest.raises(ValueError):
        tsaf.serialize(tsaf.Document((3, 3), obj))


def _location_db(tmp_path, entries, title_keys=()):
    """Build a throwaway library DB with localMediaItemLocations records,
    plus a minimal titleID record for each key in ``title_keys``."""
    db_path = tmp_path / "MediaLibrary.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE "database2" (rowid INTEGER PRIMARY KEY, collection CHAR, '
        "key CHAR, data BLOB, metadata BLOB)"
    )
    # djay's real schema: put()'s ON CONFLICT(collection, key) depends on it
    conn.execute('CREATE UNIQUE INDEX "true_primary_key" ON "database2" ("collection", "key")')
    for key, url in entries:
        o = tsaf.Obj("ADCMediaItemLocation")
        o.fields = [("uuid", key), ("sourceURIs", tsaf.Arr(tsaf.TAG_ARRAY_B, [tsaf.Url(url)]))]
        conn.execute("INSERT INTO database2 (collection,key,data) VALUES (?,?,?)",
                     ("localMediaItemLocations", key, tsaf.serialize(tsaf.Document((3, 3), o))))
    for key in title_keys:
        o = tsaf.Obj("ADCMediaItemTitleID")
        o.fields = [("uuid", key), ("title", key)]
        conn.execute("INSERT INTO database2 (collection,key,data) VALUES (?,?,?)",
                     ("mediaItemTitleIDs", key, tsaf.serialize(tsaf.Document((3, 3), o))))
    conn.commit()
    conn.close()
    return db_path


def test_find_track_requires_exact_path(tmp_path):
    """A basename that is a substring of another track's path must not match;
    only the exact source URL counts."""
    db = djaydb.DjayDB(_location_db(tmp_path, [
        ("right", "file:///m/move.opus"),
        ("wrong", "file:///m/remove.opus"),
    ]))
    assert db.find_track_by_path("/m/move.opus") == "right"
    assert db.find_track_by_path("/m/remove.opus") == "wrong"
    assert db.find_track_by_path("/m/missing.opus") is None
    db.close()


def test_find_track_encoded_and_unicode(tmp_path):
    """djay stores percent-encoded URLs, and accented names may differ in
    Unicode normalization (NFC vs NFD) from the query path — both must match."""
    import unicodedata
    from urllib.parse import quote

    nfc = "/m/Sub Dir/Natty Lou - Gautiér.opus"           # é as one codepoint
    nfd = unicodedata.normalize("NFD", nfc)                # é as e + combining accent
    db = djaydb.DjayDB(_location_db(tmp_path, [
        ("spaced", "file:///m/" + quote("Sub Dir/Jend - Darling.mp3")),
        ("accent", "file://" + quote(nfd)),                # stored NFD, queried NFC
    ]))
    assert db.find_track_by_path("/m/Sub Dir/Jend - Darling.mp3") == "spaced"
    assert db.find_track_by_path(nfc) == "accent"
    db.close()


def test_find_track_ignores_phantom_duplicate(tmp_path):
    """A leftover location record with no other metadata must lose to the one
    real catalog entry (the key that has a titleID record)."""
    db = djaydb.DjayDB(_location_db(tmp_path, [
        ("real", "file:///m/dup.opus"),
        ("phantom", "file:///m/dup.opus"),
    ], title_keys=["real"]))
    assert db.find_track_by_path("/m/dup.opus") == "real"
    db.close()


def test_find_track_refuses_ambiguous(tmp_path):
    """Two entries that are both real (or both phantom) stay ambiguous."""
    for titled in ([], ["one", "two"]):
        db = djaydb.DjayDB(_location_db(tmp_path / str(len(titled)), [
            ("one", "file:///m/dup.opus"),
            ("two", "file:///m/dup.opus"),
        ], title_keys=titled))
        with pytest.raises(ValueError, match="ambiguous"):
            db.find_track_by_path("/m/dup.opus")
        db.close()
