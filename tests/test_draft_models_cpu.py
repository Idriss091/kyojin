"""CPU tests of the DFlash / EAGLE-3 reference drafters and the speculative loop (exllamav3/generator/draft_models.py).

Tiny random-weight target, torch CPU only, no exllamav3 extension. Greedy exactness: spec output == plain greedy output.
"""
import importlib.util
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("draft_models", os.path.join(ROOT, "exllamav3/generator/draft_models.py"))
dm = importlib.util.module_from_spec(spec)
sys.modules["draft_models"] = dm
spec.loader.exec_module(dm)

H, V, NL, NH, NKV, HD, FF = 32, 50, 4, 4, 2, 8, 64
TAPS = [0, 1, 3]            # export indices (outputs of these target layers)


def rnd(g, *shape, s = 0.2):
    return torch.randn(*shape, generator = g) * s


class TinyTarget:
    """Llama-like target with KV cache, exports the outputs of layers TAPS."""

    def __init__(self, seed = 0):
        g = torch.Generator().manual_seed(seed)
        self.embed, self.head, self.norm = rnd(g, V, H, s = 1.0), rnd(g, V, H, s = 1.0), 1 + rnd(g, H, s = 0.1)
        self.layers = [{
            "q": rnd(g, NH * HD, H), "k": rnd(g, NKV * HD, H), "v": rnd(g, NKV * HD, H), "o": rnd(g, H, NH * HD),
            "gate": rnd(g, FF, H), "up": rnd(g, FF, H), "down": rnd(g, H, FF),
            "n1": 1 + rnd(g, H, s = 0.1), "n2": 1 + rnd(g, H, s = 0.1),
        } for _ in range(NL)]
        self.inv = dm._inv_freq(HD, 10000.0, None)
        self.kv = [[None, None] for _ in range(NL)]
        self.len = 0

    def forward(self, ids):
        ids = torch.tensor(ids)
        n = ids.numel()
        pos = torch.arange(self.len, self.len + n)
        x, taps = self.embed[ids], []
        for i, L in enumerate(self.layers):
            h = dm._rms(x, L["n1"], 1e-6)
            q = dm._rope((h @ L["q"].T).view(n, NH, HD), pos, self.inv)
            k = dm._rope((h @ L["k"].T).view(n, NKV, HD), pos, self.inv)
            v = (h @ L["v"].T).view(n, NKV, HD)
            kp, vp = self.kv[i]
            kk = k if kp is None else torch.cat((kp, k))
            vv = v if vp is None else torch.cat((vp, v))
            self.kv[i] = [kk, vv]
            tot = kk.shape[0]
            mask = torch.arange(tot)[None] <= (tot - n + torch.arange(n))[:, None]
            x = x + dm._attend(q, kk, vv, mask) @ L["o"].T
            m = dm._rms(x, L["n2"], 1e-6)
            x = x + (torch.nn.functional.silu(m @ L["gate"].T) * (m @ L["up"].T)) @ L["down"].T
            if i in TAPS:
                taps.append(x)
        self.len += n
        return dm._rms(x, self.norm, 1e-6) @ self.head.T, torch.cat(taps, -1)

    def truncate(self, n):
        self.len = n
        self.kv = [[k[:n], v[:n]] if k is not None else [None, None] for k, v in self.kv]


def plain_greedy(seed, prompt, n):
    t = TinyTarget(seed)
    logits, _ = t.forward(prompt)
    out = [int(logits[-1].argmax())]
    while len(out) < n:
        logits, _ = t.forward([out[-1]])
        out.append(int(logits[-1].argmax()))
    return out


def eagle_weights(seed, dv = 32, qk_norm = False):
    g = torch.Generator().manual_seed(seed)
    w = {
        "fc.weight": rnd(g, H, len(TAPS) * H),
        "midlayer.input_layernorm.weight": 1 + rnd(g, H, s = 0.1), "midlayer.hidden_norm.weight": 1 + rnd(g, H, s = 0.1),
        "midlayer.self_attn.q_proj.weight": rnd(g, NH * HD, 2 * H), "midlayer.self_attn.k_proj.weight": rnd(g, NKV * HD, 2 * H),
        "midlayer.self_attn.v_proj.weight": rnd(g, NKV * HD, 2 * H), "midlayer.self_attn.o_proj.weight": rnd(g, H, NH * HD),
        "midlayer.post_attention_layernorm.weight": 1 + rnd(g, H, s = 0.1),
        "midlayer.mlp.gate_proj.weight": rnd(g, FF, H), "midlayer.mlp.up_proj.weight": rnd(g, FF, H),
        "midlayer.mlp.down_proj.weight": rnd(g, H, FF),
        "norm.weight": 1 + rnd(g, H, s = 0.1), "lm_head.weight": rnd(g, dv, H, s = 1.0),
        "d2t": torch.randint(0, V - dv, (dv,), generator = g),      # target id = draft id + d2t[draft id]
    }
    if qk_norm:
        w["midlayer.self_attn.q_norm.weight"] = 1 + rnd(g, HD, s = 0.1)
        w["midlayer.self_attn.k_norm.weight"] = 1 + rnd(g, HD, s = 0.1)
    return w


EAGLE_CFG = {
    "architectures": ["LlamaForCausalLMEagle3"], "hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV,
    "head_dim": HD, "intermediate_size": FF, "vocab_size": V, "draft_vocab_size": 32, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "eagle_config": {"eagle_aux_hidden_state_layer_ids": TAPS, "tap_shift": 0},
}


def eagle_drafter(seed = 1, qk_norm = False):
    t = TinyTarget(0)
    meta = dm.read_eagle3_meta(EAGLE_CFG, NL, qk_norm)
    return dm.Eagle3Ref(meta, eagle_weights(seed, qk_norm = qk_norm), t.embed)


def dflash_files(d, seed = 2):
    g = torch.Generator().manual_seed(seed)
    w = {"fc.weight": rnd(g, H, len(TAPS) * H), "hidden_norm.weight": 1 + rnd(g, H, s = 0.1), "norm.weight": 1 + rnd(g, H, s = 0.1)}
    for i in range(2):
        p = f"layers.{i}."
        w.update({
            p + "input_layernorm.weight": 1 + rnd(g, H, s = 0.1), p + "post_attention_layernorm.weight": 1 + rnd(g, H, s = 0.1),
            p + "self_attn.q_proj.weight": rnd(g, NH * HD, H), p + "self_attn.k_proj.weight": rnd(g, NKV * HD, H),
            p + "self_attn.v_proj.weight": rnd(g, NKV * HD, H), p + "self_attn.o_proj.weight": rnd(g, H, NH * HD),
            p + "self_attn.q_norm.weight": 1 + rnd(g, HD, s = 0.1), p + "self_attn.k_norm.weight": 1 + rnd(g, HD, s = 0.1),
            p + "mlp.gate_proj.weight": rnd(g, FF, H), p + "mlp.up_proj.weight": rnd(g, FF, H), p + "mlp.down_proj.weight": rnd(g, H, FF),
        })
    cfg = {"architectures": ["DFlashDraftModel"], "hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV,
           "head_dim": HD, "intermediate_size": FF, "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "block_size": 5,
           "dflash_config": {"mask_token_id": 3, "target_layer_ids": TAPS, "tap_shift": 0}}
    os.makedirs(d, exist_ok = True)
    save_file(w, os.path.join(d, "model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))


PROMPT = [5, 17, 3, 29, 8, 41, 2, 11]


def test_parse_spec():
    assert dm.parse_draft_spec("dflash:/a/b") == ("dflash", "/a/b")
    assert dm.parse_draft_spec("eagle:/a") == ("eagle", "/a")
    assert dm.parse_draft_spec("EAGLE3:/a:b") == ("eagle", "/a:b")
    assert dm.parse_draft_spec("mtp") == ("mtp", None)
    assert dm.parse_draft_spec(None) == ("mtp", None)
    assert dm.parse_draft_spec("/models/x") == (None, "/models/x")
    with pytest.raises(ValueError):
        dm.parse_draft_spec("dflash:")


def test_meta_and_taps():
    m = dm.read_eagle3_meta(EAGLE_CFG, NL)
    assert m.export_layers == TAPS and m.draft_vocab_size == 32 and m.target_hidden_size == H
    cfg = {k: v for k, v in EAGLE_CFG.items() if k != "eagle_config"}
    assert dm.read_eagle3_meta(cfg, 32).tap_layers == [2, 16, 29] and dm.read_eagle3_meta(cfg, 32).export_layers == [1, 15, 28]


def test_accept_greedy():
    assert dm.accept_greedy([1, 2, 3], [1, 2, 3, 9]) == (3, 9)
    assert dm.accept_greedy([1, 2, 3], [1, 5, 3, 9]) == (1, 5)
    assert dm.accept_greedy([1, 2], [7, 2, 3]) == (0, 7)


@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_eagle_greedy_exact(k, qk_norm):
    ref = plain_greedy(0, PROMPT, 24)
    out, st = dm.speculate_greedy(TinyTarget(0), eagle_drafter(qk_norm = qk_norm), PROMPT, 24, k)
    assert out == ref
    assert st.drafted == st.rounds * k and 0 <= st.accepted <= st.drafted
    assert sum(st.accept_hist.values()) == st.rounds and sum(a * c for a, c in st.accept_hist.items()) == st.accepted


def test_dflash_greedy_exact(tmp_path):
    dflash_files(str(tmp_path))
    t = TinyTarget(0)
    d = dm.load_reference_drafter("dflash", str(tmp_path), t.embed, t.head)
    assert d.tap_layers == TAPS
    out, st = dm.speculate_greedy(TinyTarget(0), d, PROMPT, 24, 4)
    assert out == plain_greedy(0, PROMPT, 24) and st.rounds > 0


class Scripted:
    """Drafter that proposes the target's true continuation, wrong at index `bad` (None = never wrong)."""
    kind = "scripted"
    tap_layers = TAPS

    def __init__(self, truth, plen, bad = None):
        self.truth, self.plen, self.bad = truth, plen, bad

    def new_state(self, base_pos = 0):
        return dm.DrafterState()

    def advance(self, st, feats, toks):
        st.length += feats.shape[0]
        assert feats.shape[1] == len(TAPS) * H

    def propose(self, st, token, k):
        n = st.length - self.plen + 1          # tokens generated so far
        d = [self.truth[n + j] if n + j < len(self.truth) else 0 for j in range(k)]
        if self.bad is not None:
            d[self.bad] = (d[self.bad] + 1) % V
        return d


def test_full_accept_and_partial_accept():
    truth = plain_greedy(0, PROMPT, 40)
    out, st = dm.speculate_greedy(TinyTarget(0), Scripted(truth, len(PROMPT)), PROMPT, 21, 4)
    assert out == truth[:21]
    assert st.accept_hist.get(4, 0) >= 3 and st.accepted >= 3 * 4
    out, st = dm.speculate_greedy(TinyTarget(0), Scripted(truth, len(PROMPT), bad = 2), PROMPT, 21, 4)
    assert out == truth[:21] and set(st.accept_hist) == {2}          # always 2 accepted, the rest rewound
    out, st = dm.speculate_greedy(TinyTarget(0), Scripted(truth, len(PROMPT), bad = 0), PROMPT, 15, 3)
    assert out == truth[:15] and set(st.accept_hist) == {0} and st.accepted == 0


def test_eagle_state_chunking_and_pure_propose():
    t = TinyTarget(0)
    _, feats = t.forward(PROMPT)
    d = eagle_drafter()
    a, b = d.new_state(), d.new_state()
    d.advance(a, feats, PROMPT[1:])
    d.advance(b, feats[:3], PROMPT[1:3])
    d.advance(b, feats[3:], PROMPT[3:])
    assert a.length == b.length == len(PROMPT) - 1
    assert torch.allclose(a.k, b.k, atol = 1e-6) and torch.allclose(a.v, b.v, atol = 1e-6)
    p1 = d.propose(a, 7, 5)
    assert d.propose(a, 7, 5) == p1 == d.propose(b, 7, 5) and a.length == len(PROMPT) - 1
    assert all(0 <= i < V for i in p1)


def test_loader_roundtrip(tmp_path):
    save_file(eagle_weights(1), str(tmp_path / "model.safetensors"))
    json.dump(EAGLE_CFG, open(tmp_path / "config.json", "w"))
    assert dm.detect_kind(str(tmp_path)) == "eagle"
    t = TinyTarget(0)
    d = dm.load_reference_drafter("eagle", str(tmp_path), t.embed, num_target_layers = NL)
    d0 = eagle_drafter(1)
    _, feats = t.forward(PROMPT)
    s, s0 = d.new_state(), d0.new_state()
    d.advance(s, feats, PROMPT[1:])
    d0.advance(s0, feats, PROMPT[1:])
    assert d.propose(s, 9, 4) == d0.propose(s0, 9, 4)
    assert d.d2t.shape == (32,)
    dflash_files(str(tmp_path / "df"))
    assert dm.detect_kind(str(tmp_path / "df")) == "dflash"
    (tmp_path / "x").mkdir()
    json.dump({"architectures": ["Foo"]}, open(tmp_path / "x/config.json", "w"))
    with pytest.raises(ValueError):
        dm.detect_kind(str(tmp_path / "x"))
