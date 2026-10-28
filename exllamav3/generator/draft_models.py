"""Public speculative-decoding draft models (DFlash, EAGLE-3): spec parsing, checkpoint metadata, torch reference drafters.

This file imports torch and safetensors only (no exllamav3 extension), so it runs on CPU and is the numeric oracle for the engine
path (exllamav3/architecture/eagle3.py, dflash.py). See doc/draft_models.md.

Drafter interface (both kinds):
    state = drafter.new_state(base_pos)             per sequence; base_pos = first position the drafter has features for
    drafter.advance(state, feats, toks)              feed new target features (n, n_taps * target_hidden) of consecutive positions
                                                     and the tokens that follow the earlier positions (see Eagle3Ref.advance)
    drafter.propose(state, token, k) -> list[int]    k draft ids (target vocabulary) after `token`, the last sampled token
Proposals never change the state, so a rejected draft needs no rewind. Verification is the target's multi-row forward;
`speculate_greedy` is the reference accept-while-match loop (the engine does the same in Generator.iterate_gen).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

KINDS = ("mtp", "dflash", "eagle")
_ALIASES = {"eagle3": "eagle", "eagle-3": "eagle", "mtp": "mtp", "dflash": "dflash", "eagle": "eagle"}


def parse_draft_spec(spec: str | None) -> tuple[str, str | None]:
    """'dflash:/path', 'eagle:/path' (also eagle3:), 'mtp' or a bare path (kind None = detect from the checkpoint)."""
    if not spec:
        return "mtp", None
    if spec.lower() == "mtp":
        return "mtp", None
    head, sep, rest = spec.partition(":")
    kind = _ALIASES.get(head.lower())
    if sep and kind:
        if kind != "mtp" and not rest:
            raise ValueError(f"--draft {head}: needs a path ({head}:<dir>)")
        return kind, (rest or None)
    return None, spec


def detect_kind(directory: str) -> str:
    """Kind of a drafter directory from its config / tensor names."""
    with open(os.path.join(directory, "config.json")) as f:
        cfg = json.load(f)
    arch = " ".join(cfg.get("architectures") or [])
    if "Eagle3" in arch or "eagle_config" in cfg or "draft_vocab_size" in cfg:
        return "eagle"
    if "DFlash" in arch or "dflash_config" in cfg or "block_size" in cfg:
        return "dflash"
    raise ValueError(f"{directory}: neither an EAGLE-3 nor a DFlash drafter config")


@dataclass
class Eagle3Meta:
    hidden_size: int
    target_hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int                 # target vocabulary
    draft_vocab_size: int
    rms_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tap_layers: list[int]           # target layer ids as written by the checkpoint
    norm_before_residual: bool = False
    qk_norm: bool = False
    tap_shift: int = 0              # added to tap ids to get the engine export index (output of layer j)

    @property
    def export_layers(self) -> list[int]:
        return [i + self.tap_shift for i in self.tap_layers]


def default_eagle_taps(num_target_layers: int) -> list[int]:
    """vLLM / SGLang default aux layers (2, N // 2, N - 3): they capture the INPUT of layer i = OUTPUT of layer i - 1."""
    return [2, num_target_layers // 2, num_target_layers - 3]


def read_eagle3_meta(cfg: dict, num_target_layers: int | None = None, has_qk_norm: bool = False) -> Eagle3Meta:
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    ec = cfg.get("eagle_config") or {}
    taps = ec.get("eagle_aux_hidden_state_layer_ids") or cfg.get("eagle_aux_hidden_state_layer_ids")
    if not taps:
        assert num_target_layers, "config has no eagle_aux_hidden_state_layer_ids: pass the target's layer count"
        taps = default_eagle_taps(num_target_layers)
    # The ids are "input of layer i" in the reference servers = output of layer i - 1 in the engine. Overridable per checkpoint
    shift = ec.get("tap_shift", cfg.get("tap_shift", -1))
    return Eagle3Meta(
        hidden_size = h,
        target_hidden_size = cfg.get("target_hidden_size") or h,
        num_heads = nh,
        num_kv_heads = cfg.get("num_key_value_heads", nh),
        head_dim = cfg.get("head_dim") or h // nh,
        intermediate_size = cfg["intermediate_size"],
        vocab_size = cfg["vocab_size"],
        draft_vocab_size = cfg.get("draft_vocab_size") or cfg["vocab_size"],
        rms_eps = cfg.get("rms_norm_eps", 1e-6),
        rope_theta = float(cfg.get("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))),
        rope_scaling = cfg.get("rope_scaling"),
        tap_layers = list(taps),
        norm_before_residual = bool(cfg.get("norm_before_residual", False)),
        qk_norm = has_qk_norm,
        tap_shift = int(shift),
    )


@dataclass
class DFlashMeta:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    num_layers: int
    rms_eps: float
    rope_theta: float
    rope_scaling: dict | None
    block_size: int
    mask_token_id: int
    tap_layers: list[int]
    tap_shift: int = 1              # z-lab release; checkpoints can pin it (dflash_config->tap_shift)

    @property
    def export_layers(self) -> list[int]:
        return [i + self.tap_shift for i in self.tap_layers]


def read_dflash_meta(cfg: dict) -> DFlashMeta:
    dc = cfg.get("dflash_config") or {}
    g = lambda k, d=None: dc.get(k, cfg.get(k, d))
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    return DFlashMeta(
        hidden_size = h, num_heads = nh, num_kv_heads = cfg.get("num_key_value_heads", nh),
        head_dim = cfg.get("head_dim") or h // nh, intermediate_size = cfg["intermediate_size"],
        num_layers = cfg["num_hidden_layers"], rms_eps = cfg.get("rms_norm_eps", 1e-6),
        rope_theta = float(cfg.get("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))),
        rope_scaling = cfg.get("rope_scaling"),
        block_size = g("block_size"), mask_token_id = g("mask_token_id"), tap_layers = list(g("target_layer_ids")),
        tap_shift = int(g("tap_shift", 1)),
    )


def accept_greedy(drafts: list[int], preds: list[int]) -> tuple[int, int]:
    """Accept-while-match: drafts d1..dk against the target's argmax of rows 0..k. Returns (accepted a, bonus token preds[a])."""
    a = 0
    while a < len(drafts) and preds[a] == drafts[a]:
        a += 1
    return a, preds[a]


# ---- torch reference blocks -------------------------------------------------------------------------------------------

def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + eps)).to(x.dtype) * w


def _inv_freq(head_dim, theta, scaling):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))
    if scaling and (scaling.get("rope_type") or scaling.get("type")) == "llama3":
        f, lo, hi = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
        old = scaling["original_max_position_embeddings"]
        wl = 2 * math.pi / inv
        smooth = ((old / wl) - lo) / (hi - lo)
        inv = torch.where(wl > old / lo, inv / f, torch.where(wl < old / hi, inv, (1 - smooth) * inv / f + smooth * inv))
    elif scaling and (scaling.get("rope_type") or scaling.get("type")) not in (None, "default"):
        raise NotImplementedError(f"rope scaling {scaling}")
    return inv


def _rope(x, pos, inv):
    """x (n, heads, d), neox style (rotate_half), pos (n,) absolute positions."""
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat((ang.cos(),) * 2, -1)[:, None, :], torch.cat((ang.sin(),) * 2, -1)[:, None, :]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return (x * cos + torch.cat((-x2, x1), -1) * sin).to(x.dtype)


def _attend(q, k, v, mask):
    """q (nq, Hq, d), k/v (nk, Hkv, d), mask (nq, nk) bool True = visible."""
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * q.shape[-1] ** -0.5
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v.float()).to(q.dtype).reshape(q.shape[0], -1)


def _mlp(x, w, p):
    return (F.silu(x @ w[p + "gate_proj.weight"].T) * (x @ w[p + "up_proj.weight"].T)) @ w[p + "down_proj.weight"].T


@dataclass
class DrafterState:
    base_pos: int = 0
    length: int = 0                                  # positions held (EAGLE: KV entries; DFlash: context rows)
    pending: torch.Tensor | None = None              # EAGLE: features of the last position, waiting for the next token
    k: torch.Tensor | None = None                    # EAGLE: (length, Hkv, d); DFlash: unused
    v: torch.Tensor | None = None
    ctx: torch.Tensor | None = None                  # DFlash: projected+normed context rows (length, H)
    stats: dict = field(default_factory = dict)


class Eagle3Ref:
    """EAGLE-3 chain drafter (one midlayer). Weights: fc, midlayer.*, norm, lm_head, d2t; embedding is the target's."""

    kind = "eagle"

    def __init__(self, meta: Eagle3Meta, w: dict, embed: torch.Tensor):
        self.m, self.w, self.embed = meta, w, embed
        self.inv = _inv_freq(meta.head_dim, meta.rope_theta, meta.rope_scaling)
        self.tap_layers = meta.export_layers
        self.d2t = w.get("d2t")

    # features of the target (n, n_taps * Ht) -> (n, H)
    def project(self, feats):
        return feats @ self.w["fc.weight"].T

    def _qkv(self, e, h, pos):
        m, w = self.m, self.w
        x = torch.cat((e, h), -1)
        q = (x @ w["midlayer.self_attn.q_proj.weight"].T).view(-1, m.num_heads, m.head_dim)
        k = (x @ w["midlayer.self_attn.k_proj.weight"].T).view(-1, m.num_kv_heads, m.head_dim)
        v = (x @ w["midlayer.self_attn.v_proj.weight"].T).view(-1, m.num_kv_heads, m.head_dim)
        if m.qk_norm:
            q = _rms(q, w["midlayer.self_attn.q_norm.weight"], m.rms_eps)
            k = _rms(k, w["midlayer.self_attn.k_norm.weight"], m.rms_eps)
        return _rope(q, pos, self.inv), _rope(k, pos, self.inv), v

    def _layer(self, tok, hid, pos, kpast, vpast):
        """One midlayer pass for n rows; returns (pre-norm output, k, v of these rows)."""
        m, w = self.m, self.w
        e = _rms(self.embed[tok], w["midlayer.input_layernorm.weight"], m.rms_eps)
        h = _rms(hid, w["midlayer.hidden_norm.weight"], m.rms_eps)
        res = h if m.norm_before_residual else hid
        q, k, v = self._qkv(e, h, pos)
        kk = k if kpast is None else torch.cat((kpast, k))
        vv = v if vpast is None else torch.cat((vpast, v))
        n, tot = q.shape[0], kk.shape[0]
        mask = torch.arange(tot)[None, :] <= (tot - n + torch.arange(n))[:, None]
        x = res + _attend(q, kk, vv, mask) @ w["midlayer.self_attn.o_proj.weight"].T
        x = x + _mlp(_rms(x, w["midlayer.post_attention_layernorm.weight"], m.rms_eps), w, "midlayer.mlp.")
        return x, k, v

    def _ids(self, out):
        lg = _rms(out, self.w["norm.weight"], self.m.rms_eps) @ self.w["lm_head.weight"].T
        i = lg.argmax(-1)
        return i + self.d2t[i] if self.d2t is not None else i

    def new_state(self, base_pos = 0):
        return DrafterState(base_pos = base_pos)

    def advance(self, st, feats, toks):
        """Append KV entries for the positions whose follow-up token is known. Features of consecutive positions `feats` (n, F);
        `toks` are the tokens at the positions after every feature except the last one: with no pending feature toks = ids[1:]
        of the new positions (n - 1 of them), else ids of the new positions (n). The last feature stays pending."""
        feats = feats.to(self.embed.dtype)
        f = feats if st.pending is None else torch.cat((st.pending[None], feats))
        toks = torch.as_tensor(toks, dtype = torch.long).reshape(-1)
        assert toks.numel() == f.shape[0] - 1, (toks.numel(), f.shape[0])
        if toks.numel():
            pos = torch.arange(st.base_pos + st.length, st.base_pos + st.length + toks.numel())
            _, k, v = self._entry(st, toks, f[:-1], pos)
            st.k = k if st.k is None else torch.cat((st.k, k))
            st.v = v if st.v is None else torch.cat((st.v, v))
            st.length += toks.numel()
        st.pending = f[-1].clone()

    def _entry(self, st, toks, f, pos):
        # K/V of an entry depend only on (token, projected feature): no attention over the past is needed to write them
        h = self.project(f)
        e = _rms(self.embed[toks], self.w["midlayer.input_layernorm.weight"], self.m.rms_eps)
        hn = _rms(h, self.w["midlayer.hidden_norm.weight"], self.m.rms_eps)
        _, k, v = self._qkv(e, hn, pos)
        return None, k, v

    @torch.no_grad()
    def propose(self, st, token, k):
        out, ids = self.project(st.pending[None]), []
        kk, vv = st.k, st.v
        pos0 = st.base_pos + st.length
        tok = torch.tensor([token])
        for j in range(k):
            out, kn, vn = self._layer(tok, out, torch.tensor([pos0 + j]), kk, vv)
            kk = kn if kk is None else torch.cat((kk, kn))
            vv = vn if vv is None else torch.cat((vv, vn))
            tok = self._ids(out)
            ids.append(int(tok))
        return ids


class DFlashRef:
    """DFlash block drafter: block [anchor, mask * (B - 1)] attends to projected target context and to itself (non-causal)."""

    kind = "dflash"

    def __init__(self, meta: DFlashMeta, w: dict, embed: torch.Tensor, lm_head: torch.Tensor):
        self.m, self.w, self.embed, self.lm_head = meta, w, embed, lm_head
        self.inv = _inv_freq(meta.head_dim, meta.rope_theta, meta.rope_scaling)
        self.tap_layers = meta.export_layers

    def new_state(self, base_pos = 0):
        return DrafterState(base_pos = base_pos)

    def advance(self, st, feats, toks = None):
        h = _rms(feats.to(self.embed.dtype) @ self.w["fc.weight"].T, self.w["hidden_norm.weight"], self.m.rms_eps)
        st.ctx = h if st.ctx is None else torch.cat((st.ctx, h))
        st.length = st.ctx.shape[0]

    @torch.no_grad()
    def propose(self, st, token, k):
        m, w = self.m, self.w
        B = m.block_size
        ids = torch.tensor([token] + [m.mask_token_id] * (B - 1))
        x = self.embed[ids]
        L = st.length
        cpos = torch.arange(st.base_pos, st.base_pos + L)
        bpos = torch.arange(st.base_pos + L, st.base_pos + L + B)
        mask = torch.ones(B, L + B, dtype = torch.bool)
        for i in range(m.num_layers):
            p = f"layers.{i}."
            h = _rms(x, w[p + "input_layernorm.weight"], m.rms_eps)
            def proj(t, n, heads):
                return (t @ w[p + f"self_attn.{n}_proj.weight"].T).view(-1, heads, m.head_dim)
            q = _rms(proj(h, "q", m.num_heads), w[p + "self_attn.q_norm.weight"], m.rms_eps)
            kb = _rms(proj(h, "k", m.num_kv_heads), w[p + "self_attn.k_norm.weight"], m.rms_eps)
            vb = proj(h, "v", m.num_kv_heads)
            kc = _rms(proj(st.ctx, "k", m.num_kv_heads), w[p + "self_attn.k_norm.weight"], m.rms_eps)
            vc = proj(st.ctx, "v", m.num_kv_heads)
            q, kb, kc = _rope(q, bpos, self.inv), _rope(kb, bpos, self.inv), _rope(kc, cpos, self.inv)
            a = _attend(q, torch.cat((kc, kb)), torch.cat((vc, vb)), mask)
            x = x + a @ w[p + "self_attn.o_proj.weight"].T
            x = x + _mlp(_rms(x, w[p + "post_attention_layernorm.weight"], m.rms_eps), w, p + "mlp.")
        lg = _rms(x, w["norm.weight"], m.rms_eps) @ self.lm_head.T
        return [int(t) for t in lg.argmax(-1)[1:1 + k]]


# ---- checkpoint loading -----------------------------------------------------------------------------------------------

def load_tensors(directory: str) -> dict:
    from safetensors.torch import load_file
    out = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            out.update(load_file(os.path.join(directory, f)))
    return out


def load_reference_drafter(kind: str, directory: str, embed: torch.Tensor, lm_head: torch.Tensor | None = None,
                           num_target_layers: int | None = None, dtype = torch.float32):
    """Torch reference drafter from a public checkpoint directory (config.json + *.safetensors). embed / lm_head are the target's."""
    with open(os.path.join(directory, "config.json")) as f:
        cfg = json.load(f)
    w = {k: (t if t.dtype in (torch.int64, torch.bool) else t.to(dtype)) for k, t in load_tensors(directory).items()}
    if kind == "eagle":
        meta = read_eagle3_meta(cfg, num_target_layers, "midlayer.self_attn.q_norm.weight" in w)
        return Eagle3Ref(meta, w, embed.to(dtype))
    if kind == "dflash":
        assert lm_head is not None, "DFlash shares the target's lm_head"
        return DFlashRef(read_dflash_meta(cfg), w, embed.to(dtype), lm_head.to(dtype))
    raise ValueError(kind)


# ---- reference speculative loop ---------------------------------------------------------------------------------------

@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    accept_hist: dict = field(default_factory = dict)

    @property
    def mean_accepted(self):
        return self.accepted / max(self.rounds, 1)


def speculate_greedy(target, drafter, prompt: list[int], max_new: int, k: int, eos: int | None = None):
    """Greedy speculative decoding. `target.forward(ids) -> (logits (n, V), feats (n, n_taps * Ht))` appends to the target's KV,
    `target.truncate(n)` drops positions >= n. Returns (new tokens, SpecStats). The output equals plain greedy decoding."""
    st, stats = drafter.new_state(0), SpecStats()
    logits, feats = target.forward(prompt)
    t = int(logits[-1].argmax())
    out = [t]
    drafter.advance(st, feats, prompt[1:] if drafter.kind == "eagle" else None)
    while len(out) < max_new and t != eos:
        drafts = drafter.propose(st, t, k)
        rows = [t] + drafts
        n0 = len(prompt) + len(out) - 1                       # position of t
        logits, feats = target.forward(rows)
        a, bonus = accept_greedy(drafts, [int(i) for i in logits.argmax(-1)])
        target.truncate(n0 + a + 1)
        drafter.advance(st, feats[:a + 1], rows[:a + 1] if drafter.kind == "eagle" else None)
        stats.rounds += 1
        stats.drafted += len(drafts)
        stats.accepted += a
        stats.accept_hist[a] = stats.accept_hist.get(a, 0) + 1
        out += drafts[:a] + [bonus]
        t = bonus
        if eos is not None and eos in out:
            out = out[:out.index(eos) + 1]
            break
    return out[:max_new], stats
