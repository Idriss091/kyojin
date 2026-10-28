# Public draft models: DFlash and EAGLE-3

`--draft dflash:<dir>` and `--draft eagle:<dir>` load a public drafter checkpoint (safetensors + config.json) next to the target.
No training. MTP stays the default (`--draft mtp`, heads inside the target pack). Verification is always the target's
multi-row forward, so greedy output equals plain greedy output.

Code map:

| Piece | File |
| --- | --- |
| Spec parser, checkpoint metadata, torch reference drafters, reference spec loop (CPU, no extension) | `exllamav3/generator/draft_models.py` |
| DFlash engine model (existing: any target with layer export) | `exllamav3/architecture/dflash.py`, `dflash2.py` |
| EAGLE-3 engine model | `exllamav3/architecture/eagle3.py`, `exllamav3/modules/arch_specific/eagle3.py` |
| Loader `load_draft(spec, model, cache)` | `exllamav3/generator/draft_models.py` |
| Generator loop | `Generator.iterate_draftmodel_mtp_gen` (EAGLE-3 rides on it), `iterate_draftmodel_dflash_gen` |
| CPU tests | `tests/test_draft_models_cpu.py` |

## Drafter interface

A drafter owns per-sequence state and exposes three calls (reference: `Eagle3Ref`, `DFlashRef`):

- `advance(state, feats, toks)`: feed target features of new consecutive positions, `feats` = the tapped hidden states concatenated
  on the last dim (`n_taps * target_hidden`).
- `propose(state, token, k)`: `k` draft ids (target vocabulary) after `token`, the last sampled token (not yet seen by the target).
  Proposals leave the state untouched, so a rejected draft needs no rewind.
- `tap_layers`: target layer indices whose outputs must be exported.

Round (same as MTP): `rows = [t, d1..dk]` go through one target forward (the existing R-row verify path, which also exports the taps);
accept-while-match gives `a` accepted drafts and a bonus token; the drafter then `advance`s with the first `a + 1` feature rows. In the
engine the state is the draft paged cache (`draft_cache`, same page table as the target) and "rewind" is the cache-position reset the
MTP path already does.

## Target hidden states

The engine already exports any layer outputs through `params["export_state_layers"]` (TransformerBlock, all architectures; result in
`params["export_states"]`, list of `(B, R, H)` fp16 tensors). A drafter declares its taps in `draft_verifier_params`. Tap ids in
checkpoints are not uniform:

- DFlash (z-lab): `target_layer_ids`, export index = id + `tap_shift` (default 1, pin per checkpoint with `dflash_config.tap_shift`).
- EAGLE-3: `eagle_config.eagle_aux_hidden_state_layer_ids`; vLLM/SGLang capture the INPUT of layer i, which is the output of layer
  i - 1, so export index = id - 1 (`tap_shift` default -1, overridable). Default ids when the config has none: `(2, N // 2, N - 3)`.

The tap shift is measured by acceptance on a real checkpoint (a wrong shift gives about 0 accepted per round, not wrong output).

## DFlash

Block drafter. Block = `[t, mask * (B - 1)]` embedded with the target embedding (or a shipped `mask_embedding`). Context = `fc(cat taps)`
then `hidden_norm`; every layer projects context rows to K/V (k_norm, RoPE at absolute positions) which are cached; the block's own K/V
are written at the block positions; attention over `[context, block]` is non-causal. Output: `norm`, then the TARGET's lm_head.
Draft = argmax at block positions 1..B-1. `k <= B - 1`.

Checkpoint (z-lab, `architectures: ["DFlashDraftModel"]`): config `hidden_size, num_attention_heads, num_key_value_heads, head_dim,
intermediate_size, num_hidden_layers, rms_norm_eps, rope_theta, block_size, dflash_config{mask_token_id, target_layer_ids}`.
Tensors: `fc.weight (H, n_taps*Ht)`, `hidden_norm.weight`, `layers.N.{input_layernorm, post_attention_layernorm}.weight`,
`layers.N.self_attn.{q,k,v,o}_proj.weight`, `layers.N.self_attn.{q,k}_norm.weight`, `layers.N.mlp.{gate,up,down}_proj.weight`,
`norm.weight`. No embedding, no lm_head. Variants (sliding window, sinks, bidirectional flag, mask_embedding) are handled by
`DFlashConfig`. MiMo's drafter already loads through `DFlashModel` (its variant switches live in `dflash.py`);
`tools/mimo/serve.py` and the Qwen/Nanbeige servers use that same class, so there is no second implementation to move.

## EAGLE-3

Feature-level chain drafter with ONE decoder layer (`midlayer`). Position i consumes `(token_{i+1}, feature_i)` and predicts token
i + 2:

```
h0  = fc(cat(low, mid, high taps))                  step 1; later steps h = previous midlayer output (pre-norm)
e   = input_layernorm(embed(token))  ;  hn = hidden_norm(h)
qkv = [q,k,v]_proj(cat(e, hn))                      input width 2H, RoPE at absolute position, causal over the drafter KV
x   = h + o_proj(attn)   ;   x = x + mlp(post_attention_layernorm(x))     residual is the raw h (norm_before_residual=false)
id  = argmax(lm_head(norm(x)))  -> target id = id + d2t[id]               draft vocabulary is smaller (e.g. 32000)
```

The embedding is the target's. A chain of `k` steps costs `k` single-row layer passes. Tree drafting (several candidates per step,
tree attention mask in the verify forward) is the later extension; the interface does not change, only `propose` returns a tree and
the verify mask grows.

Drafter KV: one layer, same paged-cache layout as other drafters. State is `kv_position`-aligned with the target: entries for
positions 0..m-2 are written from real target features (prefill and accepted rows); the entries of a draft chain are tentative and
overwritten by the next round's catch-up, exactly like MTP.

Checkpoint (SpecForge / yuhuili / vLLM `llama_eagle3.py`): `architectures: ["LlamaForCausalLMEagle3"]`, config `hidden_size,
num_attention_heads, num_key_value_heads, intermediate_size, vocab_size, draft_vocab_size, rms_norm_eps, rope_theta,
rope_scaling, target_hidden_size (when the target width differs), eagle_config{eagle_aux_hidden_state_layer_ids, use_aux_hidden_state,
norm_before_fc}, norm_before_residual`. Tensors: `fc.weight (H, 3*Ht)`, `midlayer.input_layernorm.weight`, `midlayer.hidden_norm.weight`,
`midlayer.self_attn.{q,k,v}_proj.weight (.., 2H)`, `midlayer.self_attn.o_proj.weight`, optional `midlayer.self_attn.{q,k}_norm.weight`
(Qwen3 drafters), `midlayer.post_attention_layernorm.weight`, `midlayer.mlp.{gate,up,down}_proj.weight`, `norm.weight`,
`lm_head.weight (draft_vocab, H)`, `d2t (int64, draft_vocab)` (offset: target = draft + d2t), `t2d` (bool, unused at inference),
`embed_tokens.weight` (optional; ignored, the target's table is used).

## Families

| Target | Drafters | Notes |
| --- | --- | --- |
| Llama 3.x | EAGLE-3 `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` and SpecForge / RedHat speculators variants; DFlash on Nanbeige/Llama running | draft vocab 32000 |
| Qwen3 / Qwen3.x | EAGLE-3 (SpecForge, AngelSlim, Tengyunw), DFlash z-lab | Qwen drafters carry q_norm/k_norm; `target_hidden_size` set |
| GLM 4.x | EAGLE-3 via SpecForge/SGLang releases; no DFlash known | verify the repo exists and the arch string |
| DeepSeek V3 / R1 | native MTP preferred; EAGLE-3 drafters exist for R1 distills | MLA targets export taps the same way |

Unverified from here (no network to the hub for tensors): exact tensor lists of the third-party Qwen/GLM/DeepSeek repos. The loader
fails with the missing key name, so a mismatch is visible at load time.

## Needs GPU validation

Engine `Eagle3Model` forward (the 2H-input attention built from explicit `Linear`s), per-checkpoint tap shift, acceptance and speed
against the reference oracle `Eagle3Ref`, CUDA-graph verify with the extra taps, prefix-cache hits (drafter cache pages of a cached
prefix must hold features).
