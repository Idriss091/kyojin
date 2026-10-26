# TurboQuant KV cache (`--kv-mode tq`)

Per KV head vector (`head_dim` values, a power of two) the cache stores

* `bits` per value: the index of the nearest Lloyd-Max centroid of `R x / ||R x||`, where `R` is an orthonormal Hadamard x random-sign rotation
  (`cache/tq.py`) and the codebook is the one of a coordinate of a uniform random unit vector in `R^head_dim` (`cache/tq_codebooks.py`),
* one fp16 scale per head vector: `||R x|| / ||centroid vector||`. This is the norm correction (the reconstruction has exactly the norm of
  the input) and the only side information: `bits + 16 / head_dim` bits per value (6.06 / 5.06 at head_dim 256), against 6.5 / 5.5 for the
  default per-32-scale cache.

K and V are both stored this way. The fp16 sink-4 + window-128 ring (KVWIN) is on by default for the tq modes (`EXL3_KVWIN=0` turns it off).
No QJL stage, no boundary-layer exceptions.

```
serve.py --k-bits 5 --v-bits 5 --kv-mode tq        # tq5 (5.06 b/value); 4..6 bits, head_dim 64 / 128 / 256 / 512
```

The default (`--kv-mode quant` or nothing) is unchanged. A pack that went through `tools/qualify` has `kv_cache.json` (`kv_mode`, `k_bits`,
`v_bits`); `serve.py` uses it when no KV flag is given.

## How it fits the engine

* The pool keeps `CacheLayer_quant`'s packed layout (32-value groups, power-of-two bit planes) in `qk` / `qv`; `sk` / `sv` hold one fp16 per
  head vector. `copy_page`, save / restore, the prefix disk cache and the KVWIN stash move the same four tensors.
* `R` is orthonormal, so `q . k = (R q) . (R k)` and `o = R^T (sum p (R v))`: `attn_dispatch` rotates q, k and v on the way in and the
  output on the way out, and the pool, the ring and every kernel work in the rotated domain (one fused Triton launch for q, k, v).
* Writer: `tq_triton._tq_write_kernel` (Triton, ROCm + CUDA). Decode: `qdec.hip` with `-DTQ=1` (codebook in LDS, one scale per head) on
  ROCm; the Triton split kernel (`_qc_load_kt/v`, `TQ`) elsewhere. Prefill: the existing staging, with `tq_dequant_window` filling the fp16
  scratch (rotated domain) before the fp16 attention kernel. The BC graph attention declines tq layers.
* `tools/qualify` (kv stage) measures the decode-faithful 32K KL against k8v8 of k6v6, tq6 and tq5 once per model and keeps the cheapest
  format whose KL is <= k6v6's (else k6v6); `--no-kv-modes` skips it.
