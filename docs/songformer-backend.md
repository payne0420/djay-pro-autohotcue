# ml-songformer structure backend

Optional structure segmentation via [SongFormer](https://github.com/ASLP-lab/SongFormer)
(ASLP-lab), an MSA model that fuses MuQ + MusicFM SSL embeddings at 30 s / 420 s windows.
Beat/downbeat tracking stays **beat_this**; SongFormer supplies labeled segments only
(intro/verse/chorus/bridge/inst/outro). Cue placement still snaps to beat_this downbeats
via `cuepolicy.py`.

## Why SongFormer

SongFormer leads HarmonixSet boundary accuracy among recent MSA models
(HR.5F **0.703** vs All-In-One **0.596** — [arXiv:2510.02797](https://arxiv.org/abs/2510.02797)).
autohotcue keeps beat_this for beats/downbeats and uses SongFormer for structure labels only.

## Model

| Item | Value |
|------|-------|
| HuggingFace ID | `ASLP-lab/SongFormer` |
| Pinned revision | `a75880ed1b7375ac71860ec6c4fc9c899cf99515` (2026-05-14) |
| Override | `AUTOHOTCUE_SONGFORMER_REVISION` |
| Loader | `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` |
| Download | ~2.8 GB one-time into the HF cache |
| License | CC BY 4.0 (code + weights) — attribution: SongFormer, ASLP-lab |

The revision pin is the supply-chain mitigation; override only when deliberately
re-validating a newer snapshot.

## Prerequisites

- `uv sync --extra songformer`
- `transformers>=4.51,<5` (transformers 5.x meta-device init breaks the bundled
  MusicFM torchaudio frontend)
- `ffmpeg` on `PATH`
- First run downloads the SongFormer weights (one-time network)

## Usage

```bash
uv sync --extra songformer
uv run autohotcue propose "/path/to/track.opus" --engine ml-songformer
```

Engines: `ml` / `ml-librosa` (librosa Laplacian, default), `ml-allin1`, `ml-songformer`,
`legacy`.

```bash
uv run autohotcue bench truth.json --engines ml,ml-allin1,ml-songformer,legacy
```

## Device

Defaults to **CPU** (`AUTOHOTCUE_SONGFORMER_DEVICE` unset).
`AUTOHOTCUE_SONGFORMER_DEVICE` also applies under bench/e2e runs — `analyze()`'s
`device` flag only steers beat tracking, same as with ml-allin1.

`AUTOHOTCUE_SONGFORMER_DEVICE=mps` is opt-in and currently **not recommended**: the
420 s-window forward needs >17 GB on the MPS shared pool and OOMs under
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` on a 32 GB machine. If you set both MPS
watermarks, `PYTORCH_MPS_LOW_WATERMARK_RATIO` must be ≤ `HIGH`.

## Runtime (this machine, M2 Max 32 GB, Python 3.13.3, 2026-06-12)

| Track | duration | CPU forward | full `propose` wall | peak RSS |
|-------|----------|-------------|---------------------|----------|
| Adam Port — Move (Extended Mix) | 352.7 s opus | 108 s | 2:19 (incl. model load + beat_this) | ~23.6 GB |

`effective_parallel_jobs` forces `jobs=1` for `ml-songformer` (model singleton; analysis
stays on the main process).

## Memory

CPU f32 forward pass peaks at **~24 GB transient RSS** per track (420 s attention windows).
On a 32 GB machine an uncapped excursion can swap-thrash the whole system.

The backend sets a kernel-enforced process allocation ceiling via
`resource.setrlimit(RLIMIT_DATA)` before model load (default **26 GB** via
`AUTOHOTCUE_SONGFORMER_MEM_GB`; set to **0** to disable). A hit surfaces as a clear
`RuntimeError` with remediation hints instead of silent swapping.

Alternatives evaluated and rejected:

| Approach | Outcome |
|----------|---------|
| MPS (`AUTOHOTCUE_SONGFORMER_DEVICE=mps`) | Needs >17 GB GPU pool; OOMs under default watermarks on 32 GB |
| bf16 on CPU | No Apple-silicon bf16 support; ~20× slower |
| 210 s windows (half default) | Corrupts section labels |
| Bundled SDPA "flash" path | Parked as future work — requires import shims; caused an unbounded allocation excursion in testing |

## Implementation notes

1. **Remote code + `SONGFORMER_LOCAL_DIR`**: `modeling_songformer` reads bundled config
   from the HF snapshot; the backend sets `SONGFORMER_LOCAL_DIR` automatically after
   `snapshot_download`.
2. **Multi-file repo import**: a direct `AutoModel` import of the multi-file repo fails;
   the backend retries after inserting the snapshot dir on `sys.path`.
3. **`msaf` stub**: remote `model.py` imports `msaf` only for eval-time `cal_metrics()`;
   the backend stubs `msaf` in `sys.modules` (`msaf` is not installable on numpy 2 /
   Python 3.13).
4. **Trailing NaN logits**: the model emits NaN logits in trailing frames (harmless
   `RuntimeWarning: invalid value encountered in divide`); the segment mapper rejects
   any non-finite boundary.
5. **TLS-intercepting firewalls**: machines with a socket-firewall override
   `SSL_CERT_FILE` in child processes; the backend re-points trust to certifi before
   the one-time download.
6. **Label mapping**: output labels `intro` / `verse` / `chorus` / `bridge` / `inst` /
   `outro`; `pre-chorus` → `verse`; `silence` segments dropped.

## Status / open threads

Both open threads have self-contained roadmap briefs:

- **Quantitative bench** (`ml` vs `ml-allin1` vs `ml-songformer`) — blocked on the
  ground-truth labeling task (`truth.json` does not exist yet); plan and decision rule in
  [docs/roadmap/03-songformer-bench.md](roadmap/03-songformer-bench.md). Until then the
  evidence is qualitative: musically sensible cue layouts on Afro House tracks (Drop on
  the first chorus, Breakdown on the bridge) plus the paper's boundary numbers.
- **Memory / speed fix via the bundled SDPA attention** — the only known route below the
  ~24 GB peak without quality loss; activation recipe, failure analysis, and acceptance
  criteria in [docs/roadmap/07-songformer-flash-memory.md](roadmap/07-songformer-flash-memory.md).
  **Off-machine work only** (a first attempt swap-thrashed a 32 GB machine). MPS support
  rides on this item; until it lands, CPU f32 with the kernel ceiling is the only
  supported configuration.
