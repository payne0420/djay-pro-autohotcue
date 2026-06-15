"""Decode-ahead pipeline for folder runs.

Background threads decode audio while the caller's thread runs beat_this
inference. For CPU-DSP engines (``ml-bass``, ``ml``, ``ml-librosa``) the
post-inference stage (``fit_grid`` + cue proposal) runs on separate daemon
threads so inference can overlap DSP; ``ml-allin1`` and ``ml-songformer`` keep
the full ``analyze_decoded`` path inline because their structure stage is
heavy/GPU. ``decode`` is an ffmpeg subprocess plus a libsndfile read, both of
which release the GIL, so decode threads give real parallelism without pickling
the (~70 MB) decoded array. The single beat_this / MPS context lives in the
caller's thread only (no model stacking). Bounded queues cap in-flight decoded
arrays: decode queue (``decode_threads + lookahead``) + one in inference +
DSP input queue (``dsp_threads + 1``) + up to ``dsp_threads`` in DSP workers.
Callers may safely touch the database between yields.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator

from autohotcue.analysis import (
    analyze_decoded,
    cues_from_beats,
    decode,
    infer_beats,
    is_cpu_dsp_engine,
)

_DONE = object()  # per-thread sentinel pushed when a worker thread finishes


def _responsive_put(q: queue.Queue, payload, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            q.put(payload, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _drain_queue(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _join_workers(workers: list[threading.Thread], stop: threading.Event,
                  *queues: queue.Queue) -> None:
    stop.set()
    while any(w.is_alive() for w in workers):
        for q in queues:
            _drain_queue(q)
        for w in workers:
            w.join(timeout=0.05)


def iter_analyze_pipelined(
    items: Iterable[tuple],
    *,
    engine: str = "ml-bass",
    nudge_beats: float = 0.0,
    device: str | None = None,
    decode_threads: int = 2,
    lookahead: int = 2,
    dsp_threads: int = 1,
) -> Iterator[tuple]:
    """Yield ``(token, path, result, error)`` as each track finishes.

    ``items`` is an iterable of ``(token, path, known_bpm)``. ``token`` is opaque
    to the pipeline and handed back unchanged so the caller can correlate a result
    with its bookkeeping (e.g. a DB key or a display index). ``result`` is the
    ``(TrackAnalysis, CueProposal)`` tuple, or ``None`` when ``error`` is set;
    ``error`` is the ``Exception`` raised by decode, inference, or DSP, or
    ``None``. Results arrive in completion order. Decode runs in ``decode_threads``
    daemon threads feeding a queue of at most ``decode_threads + lookahead``
    decoded arrays — a full queue blocks the decoders (backpressure bounds
    memory). For CPU-DSP engines with ``dsp_threads >= 1``, inference runs in
    the calling thread and ``cues_from_beats`` runs on ``dsp_threads`` daemon
    worker(s); otherwise the full ``analyze_decoded`` path runs inline on the
    caller's thread. Close the generator to tear decode (and DSP) threads down
    promptly.
    """
    decode_threads = max(1, decode_threads)
    offload = is_cpu_dsp_engine(engine) and dsp_threads >= 1

    src = iter(items)
    src_lock = threading.Lock()
    stop = threading.Event()
    decode_q: queue.Queue = queue.Queue(maxsize=decode_threads + max(0, lookahead))

    def _next_item():
        with src_lock:
            return next(src, None)

    def _decode_worker() -> None:
        try:
            while not stop.is_set():
                item = _next_item()
                if item is None:
                    return
                token, path, known_bpm = item
                try:
                    payload = (token, path, known_bpm, decode(path), None)
                except Exception as exc:
                    exc.__traceback__ = None
                    payload = (token, path, known_bpm, None, exc)
                if not _responsive_put(decode_q, payload, stop):
                    return
        finally:
            _responsive_put(decode_q, _DONE, stop)

    decode_workers = [
        threading.Thread(target=_decode_worker, name=f"decode-{i}", daemon=True)
        for i in range(decode_threads)
    ]
    for w in decode_workers:
        w.start()

    if not offload:
        yield from _iter_inline(
            decode_q, decode_workers, decode_threads, stop,
            engine=engine, nudge_beats=nudge_beats, device=device,
        )
        return

    dsp_threads = max(1, dsp_threads)
    dsp_in: queue.Queue = queue.Queue(maxsize=dsp_threads + 1)
    dsp_out: queue.Queue = queue.Queue(maxsize=dsp_threads + 1)

    def _dsp_worker() -> None:
        try:
            while not stop.is_set():
                try:
                    item = dsp_in.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is _DONE:
                    return
                token, path, y, beat, known_bpm = item
                del item
                try:
                    result = cues_from_beats(
                        y, path, beat, known_bpm=known_bpm,
                        engine=engine, nudge_beats=nudge_beats,
                    )
                    payload = (token, path, result, None)
                except Exception as exc:
                    exc.__traceback__ = None
                    payload = (token, path, None, exc)
                finally:
                    del y
                if not _responsive_put(dsp_out, payload, stop):
                    return
        finally:
            pass

    dsp_workers = [
        threading.Thread(target=_dsp_worker, name=f"dsp-{i}", daemon=True)
        for i in range(dsp_threads)
    ]
    for w in dsp_workers:
        w.start()

    outstanding = 0
    decode_finished = 0
    try:
        while decode_finished < decode_threads:
            while not stop.is_set():
                try:
                    token, path, result, err = dsp_out.get_nowait()
                except queue.Empty:
                    break
                outstanding -= 1
                yield token, path, result, err

            payload = decode_q.get()
            if payload is _DONE:
                decode_finished += 1
                continue

            token, path, known_bpm, y, err = payload
            del payload
            if err is not None:
                yield token, path, None, err
                continue
            try:
                beat = infer_beats(y, device)
            except Exception as exc:
                exc.__traceback__ = None
                del y
                yield token, path, None, exc
                continue
            if not _responsive_put(
                dsp_in, (token, path, y, beat, known_bpm), stop,
            ):
                return
            outstanding += 1
            del y

        for _ in range(dsp_threads):
            _responsive_put(dsp_in, _DONE, stop)

        while outstanding > 0 and not stop.is_set():
            token, path, result, err = dsp_out.get()
            outstanding -= 1
            yield token, path, result, err
    finally:
        _join_workers(decode_workers, stop, decode_q)
        _join_workers(dsp_workers, stop, dsp_in, dsp_out)


def _iter_inline(
    decode_q: queue.Queue,
    decode_workers: list[threading.Thread],
    decode_threads: int,
    stop: threading.Event,
    *,
    engine: str,
    nudge_beats: float,
    device: str | None,
) -> Iterator[tuple]:
    finished = 0
    try:
        while finished < decode_threads:
            payload = decode_q.get()
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
            except Exception as exc:
                exc.__traceback__ = None
                yield token, path, None, exc
    finally:
        _join_workers(decode_workers, stop, decode_q)
