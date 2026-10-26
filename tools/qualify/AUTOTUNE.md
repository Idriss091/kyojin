# Day-0 autotune (NORTH STAR pillar 3) — inventory, design, first version

`qualify --stage autotune` tunes the kernels a pack needs on this device once, writes the winners next to the pack
(`<pack>/tuning/<device>.bin`), and serve start loads them. A new head_dim / hidden size / expert count then runs on tuned kernels from the
first request with no hand work.

Prior work (grep of `git log --all`, REPORT.md, pr_bundles, LEADS 26/10): the runtime tuners and caches below (not redone), dense20's note
"an A/B must share one tune cache", `082449de7`/`27869d5a7` (shipped Triton + dense-GEMM seeds). No existing "tune per pack" step.

## 1. Inventory — who tunes, where the result lives, what keys it

| kernel family | backend | tunes at runtime? | cache | key | status |
|---|---|---|---|---|---|
| EXL3 GEMM / multi-GEMM (`exl3_gemm`, `exl3_mgemm`: dense linears, fused qkv, gate+up, MoE experts, lm_head) | CUDA + ROCm | yes: `coop_autotune.cu` times 5-6 tile shapes x SM count x concurrency at the first call | `EXLLAMAV3_TUNE_CACHE` (default `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`), 32-byte records, append-only, validated on load | FNV hash of (row class, K_in, N_out, bits, fp32-out, device index, cc, SM count, codebook) [+ bszm_in/out for mgemm], salted with `COOP_AUTOTUNE_VERSION` (4) | one global file per user; **a new shape is timed inside the first request / graph capture**; row classes 2,4,8,16,24 then 16-steps to 64 (rows 65..192 reuse 64; above 192 = reconstruct + dense GEMM) |
| dense prefill GEMM (rocBLAS solution) | ROCm | yes (`EXL3_DENSE_GEMM_TUNE`, bit-exact winners only; load-time warm-up `dense_warmup.py`, budget 90 s) | `~/.cache/exllamav3/dense_gemm_tune.txt` (`EXL3_DENSE_GEMM_TUNE_FILE`) + shipped seed `model/dense_gemm_tune_seed.txt` (466 lines, Qwen shapes, row classes 4352..8192) | (n, k, rows rounded up to 256, row pitch, out dtype) | cold cache pays minutes of screening in warm-up; seed only covers shipped models |
| Triton kernels (GDN/KDA/MLA, fla vendored) | ROCm (CUDA when used) | yes: `triton.autotune` | Triton cache + `<kernel>.autotune.json`; shipped seed `kernel_seed/triton-3.7.0-gfx1151.tar.xz` | Triton's own (source, args, version) | seed per Triton version/arch; new constexpr shapes compile + time on first use |
| CUDA dense prefill GEMM (`reconstruct` + cuBLAS) | CUDA | no (cuBLAS heuristics) | - | - | nothing to tune; at 95 % of the sustained TF roof for Qwen27 (prefill27) |
| flash-attn / paged attention / GDN scan | CUDA | no | - | - | library kernels |
| hard-coded tables / env knobs | both | no | env vars (`EXL3_MOE_TILE_N`, `EXL3_MOE_COOP_WIDE`/`KSPLIT`, `EXL3_GDN_VSPLIT_KDA`, `EXL3_KDA_*SPLIT`, `EXL3_PF_MSPLIT`, `EXL3_FUSED_MIN_ROWS`=1024, `AUTO_RECONSTRUCT_THRESHOLD` 192/144, `EXL3_GEMV_*`, graph buckets) | thresholds chosen by hand on Qwen27 / Qwen3.8-Flash / GLM (dense20, fuse4k, prefill27, glmnm WIDE8, idle4, moesd) | not auto-tuned; they are structural choices validated bit-exact and gated per model. Candidates for a later day-0 stage, each only if a kept model shows a >3 % lever (plateau rule) |
| MLX (Mac) | Metal | no: module constants (`DENSE_FUSED_MAX` 128->1024, `ROWS_NAX`, `NAX_SMALL_WEIGHTS`, `pick_cfg`, S6 tile per expert count) set per model class in macair24/26 | - | rows / expert count | Mac stage is a profile in `backends.py` when the Ultra arrives; same cache idea (file next to the pack) |

Gap: the GEMM cache is **global and untied from the pack**. Every new shape is timed live (stall inside the first requests, or in graph
capture), and there is no way to ship the result: a pack tuned on the author's box tunes again on every other box.

## 2. Design

`qualify --stage autotune` (after `meta`, before `parity`; GPU job under the backend locks; resumable like every stage):

1. `replay.py` starts the pack's real server (flags derived by `pack.serve_flags`, sessions = max of `--sessions`, ctx <= 16384) on a **fresh**
   tune cache (`EXLLAMAV3_TUNE_CACHE=<out>/tune_scratch`, `EXL3_PACK_TUNE=0`).
2. It drives a fixed sweep of row counts through `/v1/completions` with token-id prompts: prompts of 2..256 tokens (one forward of that many
   rows), `chunk + r` prompts (r-row tails), N=2..sessions concurrent decodes. Every kernel the architecture reaches in the served path
   is tuned by the engine's own autotuner: fused qkv / gate-up multi-GEMMs, MoE expert GEMMs, lm_head, graph captures — no per-arch code, any
   head_dim / hidden / expert count. (A model-less variant, `autotune.py`, tunes only the plain per-linear GEMMs from the safetensors headers; kept as
   a quick pre-pass and for the A/B, it misses the fused multi-GEMM keys.)
3. The cache the server wrote is exactly this pack's winners. It is copied to `<pack>/tuning/<device signature>.bin` with a manifest
   (`.json`: signature, sweep wall times, engine git, pack index sha); on ROCm the dense-GEMM tune lines go to `<sig>.dense.txt`.
   Signature = `<cc>-<SMs>sm-<device name>` (e.g. `sm86-82sm-nvidia-geforce-rtx-3090`).
4. Serve start (`tools/qwen/serve.py`, `tools/glm/serve.py`) calls `tune_cache.install_pack(model_dir)` before the first GEMM: records of
   the file that matches this device are merged into the live cache (existing records win, nothing overwritten). The C++ loader validates each
   record against the live candidate list, and the key carries device/cc/SMs/`COOP_AUTOTUNE_VERSION`, so a stale or foreign file only misses
   and the runtime tuner fills the gap as today (safe fallback). `EXL3_PACK_TUNE=0` disables the merge (cold-cache A/B).
5. `qualify.json` gets `tuning{tuning_dir, files, records, wall_s}`; the stage wall time goes into the report next to the other stages.

Keyed per (device, kernel version): the file name carries the device signature, the record key carries the kernel version. A kernel change
(`COOP_AUTOTUNE_VERSION` bump) invalidates by construction; re-run the stage.

Not in v1: tuning the env-knob thresholds above; MLX; MoE packs were not run (replay covers them by construction, see the Flint/Qwen3.8-Flash
follow-up in the log).
