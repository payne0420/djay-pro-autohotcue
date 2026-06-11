"""Benchmark cue placement against hand-labeled ground truth."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from autohotcue import analysis


@dataclass
class GroundTruthTrack:
    path: str
    cues: dict[str, float]


@dataclass
class SlotMetrics:
    n: int = 0
    hits_beat: int = 0
    hits_bar: int = 0
    mae_sum: float = 0.0

    def record(self, predicted: float, truth: float, beat_s: float, bar_s: float) -> None:
        self.n += 1
        err = abs(predicted - truth)
        self.mae_sum += err
        if err <= beat_s:
            self.hits_beat += 1
        if err <= bar_s:
            self.hits_bar += 1

    @property
    def mae(self) -> float:
        return self.mae_sum / self.n if self.n else 0.0

    @property
    def hit_rate_beat(self) -> float:
        return self.hits_beat / self.n if self.n else 0.0

    @property
    def hit_rate_bar(self) -> float:
        return self.hits_bar / self.n if self.n else 0.0


@dataclass
class EngineResult:
    engine: str
    slots: dict[str, SlotMetrics] = field(default_factory=dict)
    runtime_total_s: float = 0.0
    tracks_run: int = 0

    def slot(self, letter: str) -> SlotMetrics:
        if letter not in self.slots:
            self.slots[letter] = SlotMetrics()
        return self.slots[letter]


def load_ground_truth(path: str) -> list[GroundTruthTrack]:
    """Parse ground-truth JSON: {"tracks": [{"path": "...", "cues": {...}}]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[GroundTruthTrack] = []
    for entry in data.get("tracks", []):
        cues = {str(k): float(v) for k, v in entry.get("cues", {}).items()}
        out.append(GroundTruthTrack(path=str(entry["path"]), cues=cues))
    return out


def yardstick_bpm(track: analysis.TrackAnalysis, fallback: float | None) -> float:
    """BPM used for ±1 beat / ±1 bar tolerances (djay when known, else tracked)."""
    if track.djay_bpm is not None and track.djay_bpm > 0:
        return track.djay_bpm
    if fallback is not None and fallback > 0:
        return fallback
    return track.bpm


def evaluate_proposal(
    prop: analysis.CueProposal,
    truth: GroundTruthTrack,
    bpm: float,
    result: EngineResult,
) -> None:
    beat_s = 60.0 / bpm
    bar_s = 4.0 * beat_s
    for letter, t_truth in truth.cues.items():
        t_pred = prop.positions.get(letter)
        if t_pred is None:
            continue
        result.slot(letter).record(t_pred, t_truth, beat_s, bar_s)


def run_engine(
    tracks: list[GroundTruthTrack],
    engine: str,
    bpm_lookup,
    jobs: int = 1,
) -> EngineResult:
    """Analyze every labeled track with one engine and accumulate metrics."""
    result = EngineResult(engine=engine)
    worker_device = "cpu" if jobs > 1 else None

    for gt in tracks:
        if not Path(gt.path).is_file():
            continue
        bpm = bpm_lookup(gt.path)
        t0 = time.perf_counter()
        track, prop = analysis.analyze(
            gt.path,
            known_bpm=bpm,
            engine=engine,
            device=worker_device,
            jobs=jobs,
        )
        result.runtime_total_s += time.perf_counter() - t0
        result.tracks_run += 1
        evaluate_proposal(prop, gt, yardstick_bpm(track, bpm), result)
    return result


def format_results(results: list[EngineResult]) -> str:
    letters = sorted({l for r in results for l in r.slots})
    lines: list[str] = []
    hdr = f"{'slot':>4}  {'engine':>6}  {'n':>3}  {'±1 beat':>8}  {'±1 bar':>8}  {'MAE':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for letter in letters:
        for res in results:
            m = res.slots.get(letter)
            if m is None or m.n == 0:
                continue
            lines.append(
                f"{letter:>4}  {res.engine:>6}  {m.n:3d}  "
                f"{m.hit_rate_beat:8.1%}  {m.hit_rate_bar:8.1%}  {m.mae:8.3f}s"
            )
    lines.append("")
    for res in results:
        avg = res.runtime_total_s / res.tracks_run if res.tracks_run else 0.0
        lines.append(
            f"{res.engine}: {res.tracks_run} tracks, "
            f"{res.runtime_total_s:.1f}s total, {avg:.1f}s/track"
        )
    return "\n".join(lines)


def cmd_bench(args) -> None:
    from autohotcue.backends import init_worker
    from autohotcue.cli import _resolve_jobs

    tracks = load_ground_truth(args.truth_json)
    if not tracks:
        raise SystemExit("no tracks in ground truth file")

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    jobs = _resolve_jobs(args.jobs)

    db = None
    if args.library:
        from autohotcue import djaydb

        db = djaydb.DjayDB(args.library)

    def bpm_lookup(path: str) -> float | None:
        if db is None:
            return args.bpm
        try:
            key = db.find_track_by_path(path)
        except ValueError:
            return args.bpm
        if key is None:
            return args.bpm
        from autohotcue import tsaf

        doc = db.get("mediaItemAnalyzedData", key)
        if doc is not None:
            bpm = doc.root.get("bpm")
            if isinstance(bpm, tsaf.F32):
                return bpm.value
        return args.bpm

    if jobs > 1 and len(tracks) > 1:
        import concurrent.futures

        def _run_one(engine: str) -> EngineResult:
            init_worker(jobs)
            return run_engine(tracks, engine, bpm_lookup, jobs=jobs)

        results: list[EngineResult] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(engines),
            initializer=init_worker,
            initargs=(jobs,),
        ) as ex:
            futs = {ex.submit(_run_one, eng): eng for eng in engines}
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: engines.index(r.engine))
    else:
        if jobs == 1:
            init_worker(1)
        results = [run_engine(tracks, eng, bpm_lookup, jobs=jobs) for eng in engines]

    if db is not None:
        db.close()

    print(format_results(results))
