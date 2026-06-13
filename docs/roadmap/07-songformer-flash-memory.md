# 07 — SongFormer: SDPA/flash attention memory fix (off-machine)

**Goal:** cut `ml-songformer`'s ~24 GB transient RSS peak to single digits by
routing MuQ + MusicFM through their bundled memory-efficient SDPA attention,
without changing segment output. **Debug this on a ≥64 GB or disposable
machine — never on the 32 GB daily driver** (a first attempt swap-thrashed
~48 GB to NVMe; that is why this is parked).

## Why the peak exists

SongFormer analyzes 420 s windows at 25 Hz → ~10 500 attention frames. Both
SSL frontends (MuQ-large and MusicFM) run transformers' stock
`Wav2Vec2ConformerEncoder` in their default path, which **materializes the
[T, T] attention scores in f32** — observed as repeated single allocations of
4.63 GiB (one per attention invocation), stacking to a ~24 GB peak on CPU and
>17 GB on the MPS shared pool.

What's already shipped on `main` as containment (see
`docs/songformer-backend.md`): CPU-only default, jobs=1 clamp,
`RLIMIT_DATA` kernel ceiling (`AUTOHOTCUE_SONGFORMER_MEM_GB`, default 26),
clean OOM errors, and an env-gated e2e so routine pytest never triggers the
peak.

## The lead (verified June 2026, parked)

Both attention stacks already ship an SDPA implementation:

- The muq pip package (`muq/muq/modules/flash_conformer.py`) and the bundled
  HF-snapshot copy (`musicfm/modules/flash_conformer.py`) are
  **byte-identical** files: transformers' wav2vec2_conformer module with the
  naive softmax replaced by `F.scaled_dot_product_attention` (line ~696).
  Same ROPE architecture (`facebook/wav2vec2-conformer-rope-large-960h-ft`
  config), same weights layout — so numerics *should* track the naive path.
- Activation recipe (already implemented behind
  `scripts/songformer_spike.py --flash`, function `_enable_flash_attention`):
  1. MuQ: `is_flash` is read from `muq_config2.json` under
     `SONGFORMER_LOCAL_DIR` → write an overlay dir with `"is_flash": true`
     (plus a copy of `msd_stats.json`) and point the env var at it.
  2. MusicFM: `is_flash=False` is hardcoded in `modeling_songformer.py` →
     pre-import `musicfm.model.musicfm_25hz` and replace `MusicFM25Hz` with a
     subclass forcing `is_flash=True` before transformers binds the name.
  3. `from modules.flash_conformer import ...` is a research-layout top-level
     import → put the muq package's inner dir (`muq/muq`) on `sys.path`.
  4. Alias `transformers.deepspeed` → `transformers.integrations.deepspeed`
     (the flash file imports the pre-4.3x shim).
  5. If `torch.backends.cuda.sdp_kernel` is missing (removed in newer torch),
     shim it with a null context — its flags are CUDA-only anyway.

## Why it is parked

The first full `--flash` run produced an **unbounded allocation excursion**
(~48 GB into swap on a 32 GB machine) before producing any output. Root cause
undiagnosed. Suspects, in order:

1. SDPA falling back to the math backend (which *does* materialize scores) on
   this device/dtype/mask combination — check
   `torch.nn.attention.sdpa_kernel` selection and whether an `attn_mask` is
   being passed that disables the efficient kernels.
2. A CPU-side fallback op or the rotary-embedding path materializing
   something [T, T]-shaped outside the SDPA call.
3. Interaction with `inference_mode` / the conformer's relative-position
   buffers.

## Plan

1. On a big-RAM or disposable box: reproduce with
   `scripts/songformer_spike.py --device cpu --flash <track>` under a hard
   process watchdog (see memory note `feedback-ram-guard-always-on`; kill at
   a sane RSS, log allocations via `PYTORCH_CUDA_ALLOC_CONF`-equivalents /
   `torch.profiler` with `profile_memory=True`).
2. Find the materializing op; fix the call (e.g. drop the mask, force the
   mem-efficient backend via `torch.nn.attention.sdpa_kernel([...])`, or
   patch the rotary application).
3. Validate: segments must match the f32 naive baseline on the reference
   track (labels identical, boundaries within one frame ≈ 0.12 s — baseline
   methodology and the Move (Extended Mix) reference are in the spike
   script's history; regenerate with `--device cpu`, no flags).
4. Acceptance: peak RSS < 10 GB on CPU **and** ideally MPS viable under a
   0.5 watermark cap (that combination would also make it the fastest
   engine); full pytest green; flash becomes the default path with the naive
   path behind an env escape hatch; update `docs/songformer-backend.md` and
   item 03's runtime expectations.

## Notes

- Independent of items 01–06; pairs with item 03 (re-bench runtime after).
- If the muq/musicfm SDPA files diverge upstream someday, prefer vendoring
  one copy — they are identical today, which is what makes the single
  `modules` sys.path resolution safe.
- Everything here is reproducible from `scripts/songformer_spike.py` on the
  merged `main`; the experiment log (what was tried and rejected: full-f16,
  bf16-CPU, autocast both devices, 210 s windows, watermark caps 0.5–0.65)
  is summarized in `docs/songformer-backend.md`'s memory section.
