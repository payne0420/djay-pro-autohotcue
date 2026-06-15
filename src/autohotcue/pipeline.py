"""Decode-ahead pipeline for folder runs.

Background threads decode audio while the caller's thread runs inference + DSP.
``decode`` is an ffmpeg subprocess plus a libsndfile read, both of which release
the GIL, so threads give real decode parallelism without pickling the (~70 MB)
decoded array across a process boundary — it is already in this process for
inference. The single beat_this / MPS context lives in the caller's thread only
(no model stacking), and a bounded queue caps the number of decoded arrays held
in flight so a long run never grows unbounded (the no-swap RAM guard). Because the
inference/DSP stage runs inline in the consumer's thread, callers may safely touch
the database between yields.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator

from autohotcue.analysis import analyze_decoded, decode

_DONE = object()  # per-thread sentinel pushed when a decoder thread finishes


def iter_analyze_pipelined(
    items: Iterable[tuple],
    *,
    engine: str = "ml-bass",
    nudge_beats: float = 0.0,
    device: str | None = None,
    decode_threads: int = 2,
    lookahead: int = 2,
) -> Iterator[tuple]:
    """Yield ``(token, path, result, error)`` as each track finishes.

    ``items`` is an iterable of ``(token, path, known_bpm)``. ``token`` is opaque
    to the pipeline and handed back unchanged so the caller can correlate a result
    with its bookkeeping (e.g. a DB key or a display index). ``result`` is the
    ``(TrackAnalysis, CueProposal)`` tuple, or ``None`` when ``error`` is set;
    ``error`` is the ``Exception`` raised by decode or analysis, or ``None``.
    Results arrive in completion order. Decode runs in ``decode_threads`` daemon
    threads feeding a queue of at most ``decode_threads + lookahead`` decoded
    arrays — a full queue blocks the decoders (backpressure bounds memory). The
    inference/DSP stage runs in the calling thread, so DB access between yields
    stays on that thread. Close the generator (it is closed automatically on GC or
    an explicit ``.close()``) to tear the decode threads down promptly.
    """
    decode_threads = max(1, decode_threads)
    src = iter(items)
    src_lock = threading.Lock()
    stop = threading.Event()
    out: queue.Queue = queue.Queue(maxsize=decode_threads + max(0, lookahead))

    def _next_item():
        with src_lock:
            return next(src, None)

    def _put(payload) -> bool:
        # Block on a full queue (backpressure) but stay responsive to `stop`.
        while not stop.is_set():
            try:
                out.put(payload, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _decode_worker() -> None:
        try:
            while not stop.is_set():
                item = _next_item()
                if item is None:
                    return
                token, path, known_bpm = item
                try:
                    payload = (token, path, known_bpm, decode(path), None)
                except Exception as exc:  # surface a decode failure per-track
                    payload = (token, path, known_bpm, None, exc)
                if not _put(payload):
                    return
        finally:
            _put(_DONE)

    workers = [
        threading.Thread(target=_decode_worker, name=f"decode-{i}", daemon=True)
        for i in range(decode_threads)
    ]
    for w in workers:
        w.start()

    finished = 0
    try:
        while finished < decode_threads:
            payload = out.get()
            if payload is _DONE:
                finished += 1
                continue
            token, path, known_bpm, y, err = payload
            if err is not None:
                yield token, path, None, err
                continue
            try:
                result = analyze_decoded(
                    y, path, known_bpm=known_bpm, engine=engine,
                    device=device, nudge_beats=nudge_beats,
                )
                yield token, path, result, None
            except Exception as exc:  # analysis failure -> surface per-track
                yield token, path, None, exc
    finally:
        # On early break / exception, free any decoder blocked on a full queue
        # (payload or sentinel put) by draining, then join. Daemon threads make
        # this best-effort; nothing leaks past process exit.
        stop.set()
        while any(w.is_alive() for w in workers):
            try:
                while True:
                    out.get_nowait()
            except queue.Empty:
                pass
            for w in workers:
                w.join(timeout=0.05)
