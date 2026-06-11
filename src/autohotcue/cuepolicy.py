"""Pure cue-placement policy from tracked beats and structure segments."""
from __future__ import annotations

from autohotcue.analysis import CueProposal
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis, bpm_octave_ratio

HIGH_LABELS = frozenset({"chorus", "solo"})
LOW_LABELS = frozenset({"break", "bridge"})
EDGE_LABELS = frozenset({"intro", "outro"})

OUTRO_BEATS_MIN = 8


def _snap(beat: BeatAnalysis, t: float) -> float:
    return beat.nearest_downbeat(t)


def _beats_after(t: float, beats: np.ndarray) -> int:
    import numpy as np

    return int(np.sum(beats > t))


def _normalize_segments(segments: list[Segment], bar_s: float) -> list[Segment]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.start, s.end))

    merged: list[Segment] = []
    i = 0
    while i < len(segs):
        seg = segs[i]
        while i + 1 < len(segs) and (seg.end - seg.start) < bar_s:
            nxt = segs[i + 1]
            if seg.end >= nxt.start - 1e-9:
                merged_start = nxt.start
            else:
                merged_start = seg.start
            seg = Segment(
                start=merged_start,
                end=nxt.end,
                label=nxt.label,
                energy_rank=max(seg.energy_rank, nxt.energy_rank),
            )
            i += 1
        merged.append(seg)
        i += 1

    out: list[Segment] = [merged[0]]
    for seg in merged[1:]:
        prev = out[-1]
        if seg.label == prev.label:
            out[-1] = Segment(
                start=prev.start,
                end=seg.end,
                label=prev.label,
                energy_rank=max(prev.energy_rank, seg.energy_rank),
            )
        else:
            out.append(seg)
    return out


def _resolve_high_low(segments: list[Segment]) -> tuple[set[int], set[int]]:
    non_edge = [i for i, s in enumerate(segments) if s.label not in EDGE_LABELS]
    high = {i for i, s in enumerate(segments) if s.label in HIGH_LABELS}
    low = {i for i, s in enumerate(segments) if s.label in LOW_LABELS}

    if not high and non_edge:
        high.add(max(non_edge, key=lambda i: segments[i].energy_rank))
    if not low and non_edge:
        low.add(min(non_edge, key=lambda i: segments[i].energy_rank))
    return high, low


def _segment_before(segments: list[Segment], idx: int) -> int | None:
    if idx <= 0:
        return None
    return idx - 1


def _first_high_after(
    segments: list[Segment],
    high: set[int],
    after_t: float,
) -> int | None:
    for i, seg in enumerate(segments):
        if i in high and seg.start > after_t:
            return i
    return None


def _first_low_after(
    segments: list[Segment],
    low: set[int],
    after_t: float,
) -> int | None:
    for i, seg in enumerate(segments):
        if i in low and seg.start > after_t:
            return i
    return None


def _max_rank_non_edge(segments: list[Segment], after_t: float) -> int | None:
    best: int | None = None
    best_rank = -1.0
    for i, seg in enumerate(segments):
        if seg.label in EDGE_LABELS or seg.start <= after_t:
            continue
        if seg.energy_rank > best_rank:
            best_rank = seg.energy_rank
            best = i
    return best


def _min_rank_between(
    segments: list[Segment],
    after_t: float,
    before_t: float,
) -> int | None:
    best: int | None = None
    best_rank = float("inf")
    for i, seg in enumerate(segments):
        if seg.label in EDGE_LABELS:
            continue
        if seg.start <= after_t or seg.start >= before_t:
            continue
        if seg.energy_rank < best_rank:
            best_rank = seg.energy_rank
            best = i
    return best


def _apply_outro_guard(
    g_t: float,
    segments: list[Segment],
    beat: BeatAnalysis,
) -> float:
    g_t = _snap(beat, g_t)
    if _beats_after(g_t, beat.beats) >= OUTRO_BEATS_MIN:
        return g_t

    candidates = sorted(
        {seg.start for seg in segments if seg.start < g_t},
        reverse=True,
    )
    for t in candidates:
        t = _snap(beat, t)
        if _beats_after(t, beat.beats) >= OUTRO_BEATS_MIN:
            return t

    for db in reversed(beat.downbeats):
        if _beats_after(db, beat.beats) >= OUTRO_BEATS_MIN:
            return float(db)
    return g_t


def _check_monotonicity(p: CueProposal) -> None:
    """Drop cues that violate ordering; append notes instead of clamping."""
    order = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strict_before = {"C": "D", "D": "E", "E": "F"}

    def val(letter: str) -> float | None:
        return p.positions.get(letter)

    changed = True
    while changed:
        changed = False
        prev: float | None = None
        for letter in order:
            cur = val(letter)
            if cur is None:
                continue
            if prev is not None and cur < prev:
                p.notes.append(
                    f"{letter}: omitted ({cur:.2f}s < previous {prev:.2f}s; monotonicity)"
                )
                del p.positions[letter]
                if letter == "G" and "H" in p.positions:
                    del p.positions["H"]
                    p.notes.append("H: omitted (G omitted)")
                changed = True
                break
            if letter in strict_before:
                nxt_letter = strict_before[letter]
                nxt = val(nxt_letter)
                if nxt is not None and cur >= nxt:
                    p.notes.append(
                        f"{letter}: omitted ({cur:.2f}s >= {nxt_letter} {nxt:.2f}s; monotonicity)"
                    )
                    del p.positions[letter]
                    changed = True
                    break
            prev = cur


def propose_cues(
    beat: BeatAnalysis,
    structure: StructureAnalysis,
    djay_bpm: float | None = None,
) -> CueProposal:
    p = CueProposal()
    segments = _normalize_segments(structure.segments, beat.bar_s())

    if djay_bpm is not None:
        ratio = bpm_octave_ratio(beat.bpm, djay_bpm)
        if ratio > 0.02:
            p.notes.append(f"djay says {djay_bpm:.1f}, tracked {beat.bpm:.1f}")

    if len(beat.downbeats) > 0:
        a = _snap(beat, float(beat.downbeats[0]))
    elif len(beat.beats) > 0:
        a = _snap(beat, float(beat.beats[0]))
    else:
        a = 0.0
    p.positions["A"] = a

    if len(beat.downbeats) < 8 or len(segments) < 2:
        p.positions["B"] = a
        p.notes.append("track too short for structure analysis; first beat only")
        return p

    intro_end: float | None = None
    for seg in segments:
        if seg.label != "intro":
            intro_end = seg.start
            break
    b = _snap(beat, intro_end if intro_end is not None else a)
    p.positions["B"] = b

    high, low = _resolve_high_low(segments)
    non_edge = [i for i, s in enumerate(segments) if s.label not in EDGE_LABELS]

    d_idx: int | None = _first_high_after(segments, high, a)
    if d_idx is None:
        d_idx = _max_rank_non_edge(segments, a)

    if len(non_edge) < 2:
        p.notes.append("no drop detected (< 2 non-edge segments)")
        d_idx = None

    d_t: float | None = None
    if d_idx is not None:
        d_t = _snap(beat, segments[d_idx].start)
        p.positions["D"] = d_t

    if d_t is not None:
        prev_idx = _segment_before(segments, d_idx)
        if prev_idx is not None:
            c_t = _snap(beat, segments[prev_idx].start)
            if b <= c_t < d_t:
                p.positions["C"] = c_t
            else:
                p.notes.append("C (Buildup): omitted (nothing between B and D)")
        else:
            p.notes.append("C (Buildup): omitted (nothing between B and D)")

    e_idx: int | None = None
    if d_t is not None:
        e_idx = _first_low_after(segments, low, d_t)
        if e_idx is None:
            outro_start = next(
                (s.start for s in segments if s.label == "outro"),
                segments[-1].start,
            )
            e_idx = _min_rank_between(segments, d_t, outro_start)
        if e_idx is not None:
            e_t = _snap(beat, segments[e_idx].start)
            if e_t > d_t:
                p.positions["E"] = e_t
            else:
                p.notes.append("E (Breakdown): omitted (no candidate after D)")
        else:
            p.notes.append("E (Breakdown): omitted (no candidate after D)")

    f_idx: int | None = None
    e_t = p.positions.get("E")
    if e_t is not None:
        f_idx = _first_high_after(segments, high, e_t)
    elif d_idx is not None:
        # E omitted: F is the second HIGH overall (first HIGH after D).
        f_idx = _first_high_after(segments, high, segments[d_idx].start)

    if f_idx is not None:
        f_t = _snap(beat, segments[f_idx].start)
        p.positions["F"] = f_t
    elif d_t is not None:
        p.notes.append("F (2nd Drop): omitted (no candidate)")

    outro_segs = [i for i, s in enumerate(segments) if s.label == "outro"]
    if outro_segs:
        g_t = _snap(beat, segments[outro_segs[-1]].start)
    else:
        g_t = _snap(beat, segments[-1].start)

    g_t = _apply_outro_guard(g_t, segments, beat)
    if _beats_after(g_t, beat.beats) >= OUTRO_BEATS_MIN:
        p.positions["G"] = g_t
        p.positions["H"] = g_t
    else:
        p.notes.append("G/H (Outro): omitted (outro guard: < 8 beats remaining)")

    _check_monotonicity(p)
    return p
