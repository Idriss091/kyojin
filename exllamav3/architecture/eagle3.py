from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import Embedding
from ..modules.arch_specific.eagle3 import Eagle3InputLayer, Eagle3Block
from ..modules.attn import prepare_for_attn


class Eagle3Config(Config):
    """Public EAGLE-3 drafter checkpoint (SpecForge / yuhuili layout, see doc/draft_models.md). The target is the one it is attached
    to: embedding table shared, taps exported from its layers."""
    arch_string = "LlamaForCausalLMEagle3"

    def __init__(self, directory: str, model_classes: dict | None = None, **kwargs):
        super().__init__(directory, model_classes or {"text": Eagle3Model}, **kwargs)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.target_hidden_size = self.read_cfg(int, "target_hidden_size", self.hidden_size) or self.hidden_size
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.head_dim = self.read_cfg(int, "head_dim", None) or self.hidden_size // self.num_q_heads
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)
        self.vocab_size = self.read_cfg(int, "vocab_size", no_default)
        self.draft_vocab_size = self.read_cfg(int, "draft_vocab_size", None) or self.vocab_size
        self.norm_before_residual = bool(self.read_cfg(bool, "norm_before_residual", False))
        self.qk_norm = self.stc.has_tensor("midlayer.self_attn.q_norm.weight")
        # Tap ids: the checkpoint's aux layers (None = the servers' default (2, N // 2, N - 3), needs the target). They name the
        # INPUT of a layer = output of layer i - 1; the engine exports layer outputs, hence tap_shift -1 (pin per checkpoint)
        self.tap_ids = self.read_cfg(list, ["eagle_config->eagle_aux_hidden_state_layer_ids", "eagle_aux_hidden_state_layer_ids"], None)
        self.tap_shift = self.read_cfg(int, ["eagle_config->tap_shift", "tap_shift"], -1)
        self.num_taps = len(self.tap_ids) if self.tap_ids else 3
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)
        self.vision = None


class Eagle3Model(Model):
    """EAGLE-3 chain drafter. Rides on the MTP draft path of the Generator (same per-step hidden feedback, one-position prefill
    shift, catch-up of accepted positions): caps mtp_draft + eagle_draft."""
    config_class = Eagle3Config

    def __init__(self, config: Eagle3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.input_layer = Eagle3InputLayer(config, "fc", config.hidden_size, config.num_taps * config.target_hidden_size)
        self.block = Eagle3Block(
            config, "midlayer", 0, config.hidden_size, config.head_dim, config.num_q_heads, config.num_kv_heads,
            config.intermediate_size, config.rms_norm_eps, config.rope_settings, config.draft_vocab_size, config.qk_norm,
            config.norm_before_residual,
        )
        self.modules = [self.input_layer, self.block]
        self.first_block_idx = 1
        self.attn_modules = [self.block.attn]
        self.last_kv_module_idx = 1
        self.logit_layer_idx = None
        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "eagle_draft": True,
            "default_draft_size": 3,
            "autosplit_load_fwd": False,
        })
        self.attached_model = None

    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)
        assert isinstance(target.modules[0], Embedding), "EAGLE-3 needs the target's embedding as first module"
        assert target.config.hidden_size == self.config.target_hidden_size, \
            f"drafter expects target hidden size {self.config.target_hidden_size}, target has {target.config.hidden_size}"
        ids = self.config.tap_ids or [2, target.config.num_hidden_layers // 2, target.config.num_hidden_layers - 3]
        self.tap_layers = [i + self.config.tap_shift for i in ids]
        self.draft_verifier_params.update({"export_state_layers": set(self.tap_layers)})

    def target_hidden_from_states(self, states: list) -> torch.Tensor:
        """Target export_states (one tensor per tapped layer, in layer order) -> the width fc expects."""
        order = sorted(range(len(self.tap_layers)), key = lambda i: self.tap_layers[i])
        assert len(states) == len(order), f"expected {len(order)} exported states, got {len(states)}"
        # transformer.py appends exports in layer order, the checkpoint's order is the tap_ids order
        inv = [order.index(i) for i in range(len(order))]
        return torch.cat([states[j] for j in inv], dim = -1)

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        return self.block.draft_ids(state, params)

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("EAGLE-3 draft model does not have its own chat template")
