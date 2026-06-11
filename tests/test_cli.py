"""Tests for CLI path expansion (directory batch mode)."""
from __future__ import annotations

import pytest

from autohotcue.cli import _expand_paths


def test_single_file_passed_through(tmp_path):
    f = tmp_path / "track.opus"
    f.write_bytes(b"")
    assert _expand_paths(str(f)) == [str(f)]


def test_single_file_extension_not_filtered(tmp_path):
    # A file given explicitly is used as-is, whatever its extension —
    # ffmpeg decides what it can decode, not the extension list.
    f = tmp_path / "track.weird"
    f.write_bytes(b"")
    assert _expand_paths(str(f)) == [str(f)]


def test_directory_recursive_audio_only_sorted(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    audio = [
        tmp_path / "b.opus",
        tmp_path / "a.MP3",  # extension match is case-insensitive
        tmp_path / "sub" / "c.ogg",
        tmp_path / "sub" / "deeper" / "d.flac",
    ]
    other = [
        tmp_path / "cover.jpg",
        tmp_path / "notes.txt",
        tmp_path / "sub" / "playlist.m3u",
    ]
    for f in audio + other:
        f.write_bytes(b"")
    assert _expand_paths(str(tmp_path)) == sorted(str(f) for f in audio)


def test_directory_skips_hidden_files_and_dirs(tmp_path):
    (tmp_path / ".cache").mkdir()
    visible = tmp_path / "track.opus"
    visible.write_bytes(b"")
    (tmp_path / "._track.opus").write_bytes(b"")  # AppleDouble sidecar
    (tmp_path / ".hidden.mp3").write_bytes(b"")
    (tmp_path / ".cache" / "buried.mp3").write_bytes(b"")
    assert _expand_paths(str(tmp_path)) == [str(visible)]


def test_directory_with_no_audio_exits(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(SystemExit):
        _expand_paths(str(tmp_path))
