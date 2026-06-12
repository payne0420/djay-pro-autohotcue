# Roadmap

One self-contained brief per workstream, written to be executed independently
(e.g. one `/goal` session each). Suggested order and dependencies:

| # | Item | Depends on | Size |
|---|------|------------|------|
| 01 | [Ground-truth bench labels](01-ground-truth-bench-labels.md) | nothing — **do first** | S (mostly user labeling time) |
| 02 | [Placement engine redesign](02-placement-engine-redesign.md) | 01 | L — the headline item |
| 03 | [SongFormer engine bench](03-songformer-bench.md) | 01 | S |
| 04 | [Fingerprint pairing reverse-engineering](04-fingerprint-pairing-re.md) | nothing (independent) | M |
| 05 | [Splice gate: sub-60s blind spot](05-splice-gate-sub60s.md) | nothing (independent) | S |
| 06 | [Half-time parity: resolve instead of refuse](06-half-time-parity-kick-evidence.md) | nothing; nicer after 01 | S |

01 is the critical path: it unblocks 02 and 03 simultaneously and converts all
future cue-quality work from "audition in djay" to "read a bench table".

Minor backlog (no doc): GitHub Dependabot reports 1 low-severity dependency
vulnerability on the default branch — triage at
`github.com/payne0420/djay-pro-autohotcue/security/dependabot/1`.

Context docs: [grid-lock-spec.md](../grid-lock-spec.md) (shipped June 2026),
[audio-alignment-fingerprint.md](../audio-alignment-fingerprint.md).
