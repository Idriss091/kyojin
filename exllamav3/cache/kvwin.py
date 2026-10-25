"""
KVWIN: exact (fp16) sink tokens + recent window on top of a quantized KV cache.

The paged quantized pool stays complete and unchanged (save/restore, prefix disk, fork and copy_page never see
the overlay). Next to it every active job owns one RING of fp16 K/V rows per full-attention layer: S sink rows
(positions 0..S-1) and a cyclic region of Rc = W + MAX_DRAFT rows holding the last Rc positions. Attention reads
the quantized pool for positions [S, q-W] and the ring for the sinks and [q-W+1, q] (an extra partial merged by the
existing flash-decoding combine), so a speculative verify of up to MAX_DRAFT rows never evicts a live window row.

The ring is per-job state with the lifecycle of the job's recurrent state: it rides the recurrent-state slot
(Cache.get_new_state / new_from_stashed / free), is stashed with the checkpoints (RecurrentCache, tail stash,
prefix disk) and restored bitwise with them. A stash without a ring blob (written with the switch off) restores
as "cold": the ring is rebuilt from the dequantized pool the first time the job attends (fill).

EXL3_KVWIN=S:W (e.g. 4:128) turns it on for CacheLayer_quant layers of models that have recurrent state slots;
unset / 0 / empty = off, which leaves every code path bit-identical to a build without this file.
"""
from __future__ import annotations
import os
from typing import NamedTuple
import torch
from ..constants import PAGE_SIZE

ANY_ACTIVE = False   # a bound ring exists in this process: the whole-forward graph declines (it cannot fill cold rings)
MAX_DRAFT = 16     # rows one verify step writes at once (BC_Attention MAX_QLEN, the decode kernels' q_len cap)


class KVWinCfg(NamedTuple):
    S: int          # sink rows
    W: int          # window rows
    Rc: int         # cyclic rows (W + MAX_DRAFT)
    R: int          # ring rows (S + Rc)


def parse_spec(spec: str | None) -> KVWinCfg | None:
    """'S:W' -> KVWinCfg; None / '' / '0' / 'off' -> None."""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("", "0", "off", "false"):
        return None
    try:
        s, w = spec.split(":")
        S, W = int(s), int(w)
    except ValueError:
        raise ValueError(f"EXL3_KVWIN must be 'sink:window' (e.g. 4:128), got {spec!r}")
    if S < 0 or W < 1 or S + W > PAGE_SIZE * 8:
        raise ValueError(f"EXL3_KVWIN={spec}: need sink >= 0 and 1 <= window, sink + window <= {PAGE_SIZE * 8}")
    return KVWinCfg(S, W, W + MAX_DRAFT, S + W + MAX_DRAFT)


def env_cfg() -> KVWinCfg | None:
    return parse_spec(os.environ.get("EXL3_KVWIN"))


def ring_rows(pos: torch.Tensor, cfg: KVWinCfg) -> torch.Tensor:
    """Ring row of absolute position(s) pos."""
    return torch.where(pos < cfg.S, pos, cfg.S + (pos - cfg.S) % cfg.Rc)


def ring_positions(end: int, cfg: KVWinCfg) -> torch.Tensor:
    """Position held by every ring row once positions < end were written; -1 = row never written."""
    r = torch.arange(cfg.R)
    last = end - 1
    p = torch.where(r < cfg.S, r, last - (last - cfg.S - (r - cfg.S)) % cfg.Rc)
    return torch.where((r < cfg.S) & (r > last) | (r >= cfg.S) & (p < cfg.S), torch.full_like(p, -1), p)


def ov_write_ref(ov_k, ov_v, slot: int, k: torch.Tensor, v: torch.Tensor, start: int, cfg: KVWinCfg):
    """Reference writer, one sequence: k, v [q_len, kvh, hd] are positions start..start+q_len-1."""
    for t in range(k.shape[0]):
        p = start + t
        row = p if p < cfg.S else cfg.S + (p - cfg.S) % cfg.Rc
        ov_k[slot, row] = k[t]
        ov_v[slot, row] = v[t]


class KVWinCtx(NamedTuple):
    """What attn_dispatch needs for one attention call: the layer's ring tensors and the per-row slots."""
    state: "KVWinLayerState"
    slots: torch.Tensor     # [bsz] int32 on the layer's device
    chunk_k: torch.Tensor | None = None     # prefill only: the chunk's own fp16 K / V rows [bsz, q_len, kvh, hd]
    chunk_v: torch.Tensor | None = None

    @property
    def cfg(self) -> KVWinCfg:
        return self.state.cfg


class KVWinLayerState:
    """Recurrent-layer-state protocol (see SWALayerState / GDNLayerState) for the ring of one full-attention layer."""

    kvwin = True
    rewind_noop = True      # positional ring: a rewind of <= MAX_DRAFT rows just leaves stale rows to be overwritten

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.cache_id = cache_id
        self.cfg: KVWinCfg | None = None
        self.cache_layer = None
        self.device = None
        self.ov_k = None
        self.ov_v = None
        self.cold = [False] * max_batch_size

    # -- Cache.__init__ hook ---------------------------------------------------------------------------------------
    def bind(self, cache_layer):
        """Called once the cache layer of this attention exists. Inert unless EXL3_KVWIN is set and the layer is a
        plain quantized cache."""
        from .quant import CacheLayer_quant
        cfg = env_cfg()
        m = self.module
        if (cfg is None or not isinstance(cache_layer, CacheLayer_quant) or cache_layer.compand_a != 0.0 or
                m.head_dim % 32 != 0 or m.window_arg() not in (None, -1, (-1, -1)) or m.logit_softcapping):
            return
        self.cfg = cfg
        self.cache_layer = cache_layer
        global ANY_ACTIVE
        ANY_ACTIVE = True
        cache_layer.kvwin = self        # BC_Attention (graph decode) and the whole-forward graph decline such layers

    @property
    def active(self) -> bool:
        return self.cfg is not None and self.ov_k is not None

    def _shape(self):
        m = self.module
        return (self.max_batch_size, self.cfg.R, m.num_kv_heads, m.head_dim)

    # -- recurrent layer state protocol ----------------------------------------------------------------------------
    def alloc(self, device):
        self.device = device
        if self.cfg is not None:
            self.ov_k = torch.zeros(self._shape(), dtype = torch.half, device = device)
            self.ov_v = torch.zeros(self._shape(), dtype = torch.half, device = device)
            self.cold = [False] * self.max_batch_size

    def free(self):
        self.ov_k = None
        self.ov_v = None
        self.device = None

    def clear(self, idx: int):
        if self.active:
            self.ov_k[idx].zero_()
            self.ov_v[idx].zero_()
        self.cold[idx] = False

    def get_checkpoint_size(self):
        if self.cfg is None:
            return 0
        m = self.module
        return 2 * self.cfg.R * m.num_kv_heads * m.head_dim * 2

    def storage_size(self):
        if self.cfg is None:
            return 0
        return 2 * int(torch.Size(self._shape()).numel()) * 2

    def get_state_tensors(self):
        return (self.ov_k, self.ov_v) if self.active else ()

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass

    def stash(self, slot, position: int = 0, pinned: bool = False):
        if not self.active:
            return None
        if pinned:
            out = tuple(torch.empty(t.shape[1:], dtype = t.dtype, pin_memory = True) for t in (self.ov_k, self.ov_v))
            for d, s in zip(out, (self.ov_k[slot], self.ov_v[slot])):
                d.copy_(s, non_blocking = True)
            return out
        return (self.ov_k[slot].cpu(), self.ov_v[slot].cpu())

    def unstash(self, slot, stashed, position: int = 0):
        if not self.active:
            return
        if stashed is None:
            self.cold[slot] = True
            return
        k, v = stashed
        self.ov_k[slot].copy_(k)
        self.ov_v[slot].copy_(v)
        self.cold[slot] = False

    def unstash_missing(self, slot):
        """The checkpoint carries no ring (written with KVWIN off): rebuild from the pool at the first attention."""
        if self.active:
            self.cold[slot] = True

    def tp_export(self, plan):
        raise NotImplementedError("EXL3_KVWIN does not support tensor parallel loading")

    # -- fill from the quantized pool --------------------------------------------------------------------------------
    def fill(self, slot: int, block_table_row: torch.Tensor, length: int):
        """Rebuild slot's ring for a sequence of `length` positions from the dequantized pool: sink rows and the last
        W positions (all that attention reads next). block_table_row: [pages] int32."""
        from exllamav3.ext import exllamav3_ext as ext
        cfg, cl = self.cfg, self.cache_layer
        self.ov_k[slot].zero_()
        self.ov_v[slot].zero_()
        pos = torch.cat([
            torch.arange(0, min(cfg.S, length)),
            torch.arange(max(cfg.S, length - cfg.W), length),
        ])
        self.cold[slot] = False
        if pos.numel() == 0:
            return
        pos = pos.to(block_table_row.device)
        page = block_table_row.index_select(0, pos // PAGE_SIZE).long()
        tok = page * PAGE_SIZE + pos % PAGE_SIZE
        m = self.module
        dim = m.num_kv_heads * m.head_dim
        G = dim // 32
        rows = ring_rows(pos, cfg)
        for q, s, bits, dst in ((cl.qk, cl.sk, cl.k_bits, self.ov_k), (cl.qv, cl.sv, cl.v_bits, self.ov_v)):
            qr = q.reshape(-1, G * bits).index_select(0, tok).contiguous()
            sr = s.reshape(-1, G).index_select(0, tok).contiguous()
            out = torch.empty((pos.numel(), dim), dtype = torch.half, device = q.device)
            ext.dequant_cache_cont(qr, sr, out, cl.compand_a)
            dst[slot].index_copy_(0, rows, out.view(-1, m.num_kv_heads, m.head_dim))


def make_ctx(cache_layer_state: KVWinLayerState | None, params: dict, device, bsz: int) -> KVWinCtx | None:
    """Ring context of one attention call from params["recurrent_states"] (one state per batch row), or None
    when the ring is inert or the call has no job states (plain batched forward: quantized cache only)."""
    st = cache_layer_state
    if st is None or not st.active:
        return None
    rs = params.get("recurrent_states")
    if not rs or len(rs) != bsz or "recurrent_slots" not in params:
        return None
    from ..util.tensor import get_for_device
    return KVWinCtx(st, get_for_device(params, "recurrent_slots", device))


def ensure_filled(ctx: KVWinCtx | None, params: dict, block_table: torch.Tensor):
    """Rebuild the rings of cold slots (restored from a checkpoint without a ring) before they are attended."""
    if ctx is None:
        return
    st = ctx.state
    cold = [i for i, r in enumerate(params["recurrent_states"]) if st.cold[r.slot]]
    if not cold:
        return
    from ..util.tensor import get_for_device
    seqlens = get_for_device(params, "cache_seqlens", "cpu")
    for i in cold:
        st.fill(params["recurrent_states"][i].slot, block_table[i], int(seqlens[i]))
