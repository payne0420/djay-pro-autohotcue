"""Benchmark cue placement against hand-labeled ground truth."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from autohotcue import analysis

VALID_ENGINES = frozenset({"ml", "legacy"})


@dataclass
class GroundTruthTrack:
    path: str
    cues: dict[str, float]


@dataclass
class SlotMetrics:
    n: int = 0
    omitted: int = 0
    hits_beat: int = 0
    hits_bar: int = 0
    mae_sum: float = 0.0
    mae_n: int = 0

    def record(self, predicted: float, truth: float, beat_s: float, bar_s: float) -> None:
        self.n += 1
        err = abs(predicted - truth)
        self.mae_sum += err
        self.mae_n += 1
        if err <= beat_s:
            self.hits_beat += 1
        if err <= bar_s:
            self.hits_bar += 1

    def record_miss(self) -> None:
        """Count a labeled slot with no predicted cue as a miss (no MAE entry)."""
        self.n += 1
        self.omitted += 1

    @property
    def mae(self) -> float:
        return self.mae_sum / self.mae_n if self.mae_n else 0.0

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


def parse_engines(spec: str) -> list[str]:
    engines = [e.strip() for e in spec.split(",") if e.strip()]
    bad = [e for e in engines if e not in VALID_ENGINES]
    if bad:
        raise SystemExit(
            f"unknown engine(s): {', '.join(bad)} (choose from {', '.join(sorted(VALID_ENGINES))})"
        )
    if not engines:
        raise SystemExit("no engines given")
    return engines


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


def tracked_bpm(path: str) -> float:
    """Beat-track a file once for a shared tolerance yardstick."""
    from autohotcue.backends import track_beats

    y = analysis.decode(path)
    beat = track_beats(y, device="cpu")
    return beat.bpm


def build_yardstick_map(
    tracks: list[GroundTruthTrack],
    bpm_lookup,
) -> dict[str, float]:
    """One BPM yardstick per track for all engines (djay, --bpm, else tracked)."""
    yardsticks: dict[str, float] = {}
    for gt in tracks:
        if not Path(gt.path).is_file():
            continue
        bpm = bpm_lookup(gt.path)
        if bpm is not None and bpm > 0:
            yardsticks[gt.path] = bpm
            continue
        if gt.path not in yardsticks:
            yardsticks[gt.path] = tracked_bpm(gt.path)
    return yardsticks


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
            result.slot(letter).record_miss()
        else:
            result.slot(letter).record(t_pred, t_truth, beat_s, bar_s)


@dataclass
class TrackBenchResult:
    runtime_s: float
    slots: dict[str, SlotMetrics] = field(default_factory=dict)


def bench_known_bpm(
    path: str,
    bpm_lookup,
    yardsticks: dict[str, float],
) -> float:
    """BPM passed to analysis: djay/--bpm when set, else the shared yardstick."""
    bpm = bpm_lookup(path)
    if bpm is not None and bpm > 0:
        return bpm
    return yardsticks[path]


def bench_one_track(
    engine: str,
    gt: GroundTruthTrack,
    yardstick: float,
    known_bpm: float,
) -> TrackBenchResult:
    """Picklable ProcessPool entry point for one labeled track."""
    t0 = time.perf_counter()
    track, prop = analysis.analyze(
        gt.path,
        known_bpm=known_bpm,
        engine=engine,
        device="cpu",
        jobs=1,
    )
    runtime_s = time.perf_counter() - t0
    partial = EngineResult(engine=engine)
    evaluate_proposal(prop, gt, yardstick, partial)
    return TrackBenchResult(runtime_s=runtime_s, slots=partial.slots)


def _merge_track_result(result: EngineResult, track_result: TrackBenchResult) -> None:
    result.runtime_total_s += track_result.runtime_s
    result.tracks_run += 1
    for letter, metrics in track_result.slots.items():
        slot = result.slot(letter)
        slot.n += metrics.n
        slot.omitted += metrics.omitted
        slot.hits_beat += metrics.hits_beat
        slot.hits_bar += metrics.hits_bar
        slot.mae_sum += metrics.mae_sum
        slot.mae_n += metrics.mae_n


def run_engine(
    tracks: list[GroundTruthTrack],
    engine: str,
    bpm_lookup,
    yardsticks: dict[str, float],
    jobs: int = 1,
) -> EngineResult:
    """Analyze every labeled track with one engine and accumulate metrics."""
    from autohotcue.backends import init_worker

    result = EngineResult(engine=engine)
    runnable = [gt for gt in tracks if Path(gt.path).is_file()]

    if jobs > 1 and len(runnable) > 1:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=init_worker,
            initargs=(jobs,),
        ) as ex:
            futs = {
                ex.submit(
                    bench_one_track,
                    engine,
                    gt,
                    yardsticks[gt.path],
                    bench_known_bpm(gt.path, bpm_lookup, yardsticks),
                ): gt
                for gt in runnable
            }
            for fut in concurrent.futures.as_completed(futs):
                _merge_track_result(result, fut.result())
    else:
        if jobs == 1:
            init_worker(1)
        for gt in runnable:
            _merge_track_result(
                result,
                bench_one_track(
                    engine,
                    gt,
                    yardsticks[gt.path],
                    bench_known_bpm(gt.path, bpm_lookup, yardsticks),
                ),
            )
    return result


def format_results(results: list[EngineResult]) -> str:
    letters = sorted({l for r in results for l in r.slots})
    lines: list[str] = []
    hdr = (
        f"{'slot':>4}  {'engine':>6}  {'n':>3}  {'omitted':>7}  "
        f"{'±1 beat':>8}  {'±1 bar':>8}  {'MAE':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for letter in letters:
        for res in results:
            m = res.slots.get(letter)
            if m is None or m.n == 0:
                continue
            lines.append(
                f"{letter:>4}  {res.engine:>6}  {m.n:3d}  {m.omitted:7d}  "
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
    from autohotcue.cli import _resolve_jobs

    tracks = load_ground_truth(args.truth_json)
    if not tracks:
        raise SystemExit("no tracks in ground truth file")

    engines = parse_engines(args.engines)
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

    yardsticks = build_yardstick_map(tracks, bpm_lookup)
    results = [run_engine(tracks, eng, bpm_lookup, yardsticks, jobs=jobs) for eng in engines]

    if db is not None:
        db.close()

    print(format_results(results))
