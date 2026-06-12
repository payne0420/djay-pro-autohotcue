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

## Status / pending

- Quantitative bench comparison (`ml` vs `ml-allin1` vs `ml-songformer`) is blocked on
  the ground-truth labeling task (`truth.json` does not exist yet).
- Qualitative check on Afro House tracks shows musically sensible cue layouts (Drop on
  the first chorus, Breakdown on the bridge).
- MPS / fp16 / window-size performance tuning is deferred.
