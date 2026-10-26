# Checkpoint input formats the converters read

The EXL3 converters take a vendor checkpoint and requantize it. Quantized vendor inputs are dequantized to float32
exactly per the vendor spec (one linear at a time), cast to fp16 and fed to the usual pipeline.

One reader holds the format knowledge: `exllamav3/foreign_quant/vendor_input.py` (numpy only, no torch, one file; the Mac converter
carries a byte-identical copy as `kyojin_mlx/vendor_input.py`). `Linear.load_fp16` (CUDA/ROCm `convert.py`) uses a torch twin of the same
maths on the device (tests assert fp16 bit equality with the numpy reference); `tools/bigconv` and `tools/conv_mlx.py` use the numpy reader.

Check what a checkpoint is before converting (headers only, also works on a `tensor_map.json` of a remote model):

    PYTHONPATH=tools python -m bigconv formats --src CKPT_DIR [--json]
    python -m bigconv preflight ... --src CKPT_DIR          # adds an `input-format` line; FAIL on integer quants nobody reads (GPTQ / AWQ)

## Formats

| format | tensors (per linear `P`) | value | where seen |
|---|---|---|---|
| BF16 / F16 / F32 | `P.weight` | as stored | everything |
| FP8 block grid | `P.weight` F8_E4M3 + `P.weight_scale_inv` (F32 or BF16) `[ceil(out/bo), ceil(in/bi)]`, `weight_block_size` in config | `w * scale` | DeepSeek-V3, Qwen3 / Qwen3.8 FP8, GLM, MiniMax-M2 |
| FP8 per-tensor / per-channel | `P.weight` + `P.weight_scale` `[]`, `[1]`, `[out]`, `[out,1]` | `w * scale` | ModelOpt / compressed-tensors FP8 |
| FP8 E8M0 grid | `P.weight` + `P.scale` F8_E8M0 (32x32) | `w * 2^(b-127)` | DeepSeek-V4 |
| MXFP8 | `P.weight` F8_E4M3 + `P.weight_scale_inv` **U8** `[out, in/32]` | `w * 2^(b-127)` | MiniMax-M3 (MXFP8, attention of the NVFP4 mix) |
| MXFP4 linear | `P.weight` I8/U8 `[out,in/2]` + `P.scale` F8_E8M0 or `P.weight_scale` U8 `[out,in/32]` | `e2m1 * 2^(b-127)`, low nibble first | DeepSeek-V4, MiMo-V2, AMD Quark, AutoRound |
| MXFP4 gpt-oss blocks | `P_blocks` U8 `[...,G,16]` + `P_scales` U8 `[...,G]` (fused expert tensors) | same, 32 values per block | gpt-oss |
| NVFP4 ModelOpt | `P.weight` U8 `[out,in/2]`, `P.weight_scale` F8_E4M3 `[out,in/16]`, `P.weight_scale_2` F32 `[]` | `(e2m1 * scale) * scale_2` | NVIDIA ModelOpt (Qwen3.8-Flash-Next-NVFP4, MiniMax-M3-NVFP4) |
| NVFP4 compressed-tensors | `P.weight_packed`, `P.weight_scale`, `P.weight_global_scale` F32 `[1]` | `(e2m1 * scale) / global` | RedHatAI, llm-compressor |

`input_scale` / `input_global_scale` (activation scales) are ignored. Not read: integer quants (GPTQ / AWQ / bitsandbytes) and
GGUF (the latter is served as shipped, see `exllamav3/foreign_quant`); the detector reports them instead of guessing. Tensors that are not
linears (embeddings, norms, routers, conv1d) must be dense in the source; every flagship checkpoint inspected keeps them BF16/F32.

Rounding: E2M1 x E4M3 / E8M0 products are exact in fp32, so each format rounds at most once (the NVFP4 global scale, or an fp32 / BF16 block scale).

## Status by converter (26/10, branch `feat/convin` on v1.7.4; Mac `feat/convin-mlx`)

Legend: R reads, D dequantizes to the spec (bit-exact against an independent reference), S streams one linear at a time, T tested.
`-` = not supported.

| format | `convert.py` + `Linear` (CUDA / ROCm) | `tools/bigconv` | Mac `conv_mlx` |
|---|---|---|---|
| BF16 / F16 / F32 | R D S T | plan + reader: R D S T | R D S T |
| FP8 block, `weight_scale_inv`, divisible | R D S T (was R, untested) | R D S T | R D S T |
| FP8 block, non divisible (ceil-div) or `weight_scale` grid | R D S T (was: assert / wrong shape) | R D S T | R D S T |
| FP8 per-tensor / per-channel | R D S T | R D S T | R D S T |
| FP8 E8M0 grid (`.scale`) | R D S T (DeepSeek-V4 route) | R D S T | R D S T |
| MXFP8 (U8 `weight_scale_inv`) | R D S T (was: assert) | R D S T | R D S T |
| MXFP4 linear (U8 / E8M0 scale) | R D S T | R D S T | R D S T |
| MXFP4 gpt-oss blocks (fused experts) | R D S T (path existed, now tested) | reader R D T | - (not a Qwen3.5 layout) |
| NVFP4 ModelOpt | R D S T (was: not read) | R D S T | R D S T |
| NVFP4 compressed-tensors | R D S T (was: not read) | R D S T | R D S T |
| integer quants (GPTQ / AWQ) | - (refused) | preflight FAIL | refused (SystemExit) |

Limits: bigconv's capture forward only exists for the DeepSeek-V4.1 runner (own loader in `exllamav3.dsv41`); other adapters get the
classification (`Cls.fmt`, scale and global-scale names, packed `in_f` x2) and `bigconv.formats.read_linear`, and plug the reader into their
capture forward when written. Fused block-scaled FP8 expert tensors (one 3D tensor per projection) are read only with a per-expert
scalar scale (Mistral-Small-4 layout); every checkpoint checked ships per-expert 2D tensors.

## Tests

CPU only: `tests/test_vendor_input_cpu.py` (codebooks against torch and ml_dtypes, FP8 / MXFP4 / NVFP4 synthetic checkpoints with vendor
tensor names, bit-exact vs a float64 numpy reference and a numpy encoder round trip), `tests/test_linear_vendor_load_cpu.py` (the
`Linear` loader on the same synthetic checkpoints, fp16 bit equality), `tests/bigconv/test_formats.py`. Mac: `tests/test_vendor_src.py`
and `tests/test_conv_vendor_e2e.py` (tiny MoE converted from a vendor source gives the same pack as from its own float16 dequant).
