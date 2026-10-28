from __future__ import annotations
from typing_extensions import override
import torch
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm, Attention, GatedMLP
from ...util.device_copy import to_device
from ...util.tensor import get_for_device


class Eagle3InputLayer(Module):
    """token ids + target features -> (B, S, 2H) = cat(target embedding, hidden). The hidden is fc(cat taps) when it arrives with
    the tap width, else it is the drafter's own previous state (chain step >= 2) and passes through."""

    def __init__(self, config: Config, key: str, hidden_size: int, tap_width: int):
        super().__init__(config, key, None)
        self.module_name = "Eagle3InputLayer"
        self.hidden_size = hidden_size
        self.tap_width = tap_width
        self.fc = Linear(config, key, tap_width, hidden_size, out_dtype = torch.half, pad_to = 1)
        self.register_submodule(self.fc)
        self.attached_model = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        raise NotImplementedError()

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        hid = params.get("target_hidden")
        assert hid is not None, "EAGLE-3 needs target_hidden"
        assert hid.shape[:-1] == x.shape, f"EAGLE-3 token/state shape mismatch: {tuple(x.shape)} vs {tuple(hid.shape)}"
        hid = get_for_device(params, "target_hidden", self.device)
        if hid.shape[-1] == self.tap_width and self.tap_width != self.hidden_size:
            hid = self.fc.forward(hid.half(), params, out_dtype = torch.half)
        emb = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        return torch.cat((to_device(emb, hid.device).half(), hid.half()), dim = -1)


class Eagle3Block(Module):
    """The single EAGLE-3 midlayer plus the draft head. The head (norm, lm_head, d2t) is kept as submodules so the loader places
    it, but forward() stops before it: the pre-norm output is both the next chain step's hidden and what sample_from_state reads."""

    def __init__(self, config, key: str, layer_idx: int, hidden_size: int, head_dim: int, num_q_heads: int, num_kv_heads: int,
                 intermediate_size: int, rms_norm_eps: float, rope_settings, draft_vocab: int, qk_norm: bool,
                 norm_before_residual: bool):
        super().__init__(config, key, None)
        self.module_name = "Eagle3Block"
        self.hidden_size = hidden_size
        self.norm_before_residual = norm_before_residual
        k = key
        norm = lambda name: RMSNorm(config = config, key = f"{k}.{name}", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
        self.input_layernorm = norm("input_layernorm")
        self.hidden_norm = norm("hidden_norm")
        self.post_norm = norm("post_attention_layernorm")
        lin = lambda name, o: Linear(config, f"{k}.self_attn.{name}", 2 * hidden_size, o, pad_to = 1)
        self.attn = Attention(
            config = config, key = f"{k}.self_attn", layer_idx = layer_idx, hidden_size = hidden_size, head_dim = head_dim,
            num_q_heads = num_q_heads, num_kv_heads = num_kv_heads, rope_settings = rope_settings,
            q_proj = lin("q_proj", num_q_heads * head_dim), k_proj = lin("k_proj", num_kv_heads * head_dim),
            v_proj = lin("v_proj", num_kv_heads * head_dim), key_o = "o_proj", out_dtype = torch.float,
            q_norm = RMSNorm(config = config, key = f"{k}.self_attn.q_norm", rms_norm_eps = rms_norm_eps) if qk_norm else None,
            k_norm = RMSNorm(config = config, key = f"{k}.self_attn.k_norm", rms_norm_eps = rms_norm_eps) if qk_norm else None,
        )
        self.mlp = GatedMLP(
            config = config, key = f"{k}.mlp", hidden_size = hidden_size, intermediate_size = intermediate_size,
            key_up = "up_proj", key_gate = "gate_proj", key_down = "down_proj", interm_dtype = torch.half, out_dtype = torch.float,
        )
        self.head_norm = RMSNorm(config = config, key = "norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
        self.lm_head = Linear(config, "lm_head", hidden_size, draft_vocab, out_dtype = torch.half, pad_to = 1)
        for m in (self.input_layernorm, self.hidden_norm, self.post_norm, self.attn, self.mlp, self.head_norm, self.lm_head):
            self.register_submodule(m)
        self.d2t = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.config.stc.has_tensor("d2t"):
            self.d2t = self.config.stc.get_tensor("d2t", device, no_defer = True).long().view(-1)

    @override
    def unload(self):
        self.d2t = None
        super().unload()

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        H = self.hidden_size
        emb, hid = x[..., :H], x[..., H:]
        e = self.input_layernorm.forward(emb, params, out_dtype = torch.half)
        hn = self.hidden_norm.forward(hid, params, out_dtype = torch.half)
        res = (hn if self.norm_before_residual else hid).float()
        res = res + self.attn.forward(torch.cat((e, hn), dim = -1), params).float()
        m = self.post_norm.forward(res, params, out_dtype = torch.half)
        res = res + self.mlp.forward(m, params).float()
        return res.clamp_(-65504.0, 65504.0).half()

    def draft_ids(self, state: torch.Tensor, params: dict, limit: int | None = None) -> torch.Tensor:
        logits = self.lm_head.forward(self.head_norm.forward(state, params, out_dtype = torch.half), params).float()
        conf, ids = torch.max(logits, dim = -1)
        if params.get("export_draft_conf"):
            params["draft_conf"] = conf
        params["draft_p"] = (1.0 / torch.exp(logits - conf.unsqueeze(-1)).sum(-1)).reshape(-1)
        if self.d2t is not None:
            ids = ids + self.d2t[ids]
        return ids
