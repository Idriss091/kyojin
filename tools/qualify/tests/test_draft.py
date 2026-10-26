"""CPU tests of the draft-path discovery (drafter.py), the draft rule and the draft A/B arithmetic."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import drafter  # noqa: E402
import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import rules  # noqa: E402
import stages  # noqa: E402
from test_plan import args, fake_pack  # noqa: E402


def fake_drafter(d: Path, hidden=128, taps=(1, 3), arch="DFlash2DraftModel", tap_shift=None, vocab=None):
    d.mkdir(parents=True, exist_ok=True)
    dc = {"block_size": 8, "target_layer_ids": list(taps)}
    if tap_shift is not None:
        dc["tap_shift"] = tap_shift
    cfg = {"architectures": [arch], "hidden_size": hidden, "dflash_config": dc}
    if vocab:
        cfg["vocab_size"] = vocab
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"x")
    return d


class DrafterTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        (self.root / "my-model-33b-exl3-4.0").mkdir()
        self.pack = fake_pack(self.root / "my-model-33b-exl3-4.0")
        self.meta = packmod.read_pack(self.pack)

    def tearDown(self):
        self.t.cleanup()

    def test_stem_strips_the_quantization_suffix(self):
        self.assertEqual(drafter.stem("nanbeige43-33b-exl3-4.0"), "nanbeige43-33b")
        self.assertEqual(drafter.stem("qwen38-27b-exl3-td-sc4.0"), "qwen38-27b")
        self.assertEqual(drafter.stem("Foo-7B-4.0bpw"), "Foo-7B")
        self.assertIn(drafter.norm("nanbeige43-33b"), drafter.norm("Nanbeige4.3-33B-DFlash2"))

    def test_structural_validation(self):
        ok = fake_drafter(self.root / "ok")
        self.assertEqual(drafter.validate(ok, self.meta), [])
        self.assertTrue(drafter.validate(fake_drafter(self.root / "w", hidden=64), self.meta))
        self.assertTrue(drafter.validate(fake_drafter(self.root / "t", taps=(1, 9)), self.meta))      # layer 9 >= 4 layers
        self.assertTrue(drafter.validate(fake_drafter(self.root / "a", arch="Qwen3ForCausalLM"), self.meta))
        self.assertTrue(drafter.validate(fake_drafter(self.root / "v", vocab=999), self.meta))
        self.assertEqual(drafter.validate(self.root / "nowhere", self.meta), ["no readable config.json"])

    def test_find_local_prefers_the_explicit_export_and_pack_dir(self):
        a = fake_drafter(self.root / "dr" / "a")
        b = fake_drafter(self.root / "dr" / "b", tap_shift=0)
        old = drafter.DRAFTER_HOME
        drafter.DRAFTER_HOME = self.root / "dr"
        try:
            self.assertEqual(drafter.find_local(self.meta)[0], str(b))               # states its tap convention
            self.assertEqual(drafter.find_local(self.meta, str(a))[0], str(a))       # explicit --drafter wins
            fake_drafter(self.pack / "dflash")                                       # a pack-side dir is a candidate too; the stated export still wins
            self.assertEqual(drafter.find_local(self.meta)[0], str(b))
        finally:
            drafter.DRAFTER_HOME = old

    def test_pick_hf_skips_quantized_variants(self):
        ids = ["x/M-DFlash2-GGUF", "x/M-DFlash2", "y/M-DFlash-long"]
        self.assertEqual(drafter.pick_hf(ids), "x/M-DFlash2")
        self.assertIsNone(drafter.pick_hf(["x/M-DFlash2-GGUF", "x/M-DFlash-mlx"]))


class RuleTests(unittest.TestCase):
    def test_adaptive_drafter_kept_when_chat_holds_and_agentic_gains(self):
        self.assertEqual(rules.choose_draft({"code": 1.23, "chat": 1.0, "agentic": 2.28})[0], True)

    def test_chat_loss_or_no_gain_refuses(self):
        self.assertEqual(rules.choose_draft({"code": 1.5, "chat": 0.94, "agentic": 2.3})[0], False)
        self.assertEqual(rules.choose_draft({"code": 1.05, "chat": 1.0, "agentic": 1.08})[0], False)
        self.assertEqual(rules.choose_draft({})[0], False)

    def test_speedups_are_means_over_both_orders(self):
        runs = [dict(variant="plain", tps={"chat": [10.0, 10.0]}), dict(variant="dflash", tps={"chat": [20.0, 20.0]}),
                dict(variant="dflash", tps={"chat": [22.0, 22.0]}), dict(variant="plain", tps={"chat": [12.0, 12.0]})]
        self.assertEqual(stages.speedups(runs), {"chat": 1.909})
        self.assertEqual(stages.draft_variants(), ["plain", "dflash", "dflash", "plain"])


class PlanOrderTests(unittest.TestCase):
    def test_draft_runs_before_autotune_and_maxctx(self):
        o = planmod.ORDER
        self.assertLess(o.index("kv"), o.index("draft"))
        self.assertLess(o.index("draft"), o.index("autotune"))
        self.assertLess(o.index("autotune"), o.index("maxctx"))

    def test_serve_flags_carry_the_drafter(self):
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
            f = packmod.serve_flags(m, port=1, ctx=4096, sessions=1, k_bits=6, v_bits=6, draft="dflash", dflash_dir="/d")
            self.assertEqual((f[f.index("--draft-policy") + 1], f[f.index("--dflash-dir") + 1]), ("dflash", "/d"))
            self.assertNotIn("--dflash-dir", packmod.serve_flags(m, port=1, ctx=4096, sessions=1, k_bits=6, v_bits=6))


class LaneTests(unittest.TestCase):
    def test_lane_command_and_hermes_block_come_from_the_measured_config(self):
        import report
        with tempfile.TemporaryDirectory() as t:
            m = packmod.read_pack(fake_pack(Path(t)))
            a = args(sessions=[1, 3])
            ctxs = {"1": 65536, "3": 49152}
            d = {"draft": dict(policy="dflash", dflash_dir="/d"), "kv": dict(mode="quant", kvwin="4:128")}
            ln = report.lane(m, a, ctxs, [6, 6], d)
            self.assertEqual((ln["sessions"], ln["ctx"]), (1, 65536))                   # 49152/3 = 16K < 24K share: ship one session
            a = args(sessions=[1, 3], min_share=8192)
            ln = report.lane(m, a, ctxs, [6, 6], d)
            self.assertEqual((ln["sessions"], ln["ctx"]), (3, 49152))
            self.assertIn("--dflash-dir /d", ln["cmd"]); self.assertIn("--kvwin 4:128", ln["cmd"]); self.assertIn("EXL3_PREFIX_DISK_DIR", ln["cmd"])
            h = report.hermes(a, ctxs)
            self.assertEqual((h["model_context_length"], h["per_session_share"], h["compression_threshold"]), (49152, 16384, 0.27))
            self.assertIsNone(report.lane(m, a, {"1": None}, [6, 6], {}))


if __name__ == "__main__":
    unittest.main()
