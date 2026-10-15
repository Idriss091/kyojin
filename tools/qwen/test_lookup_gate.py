"""install_lookup gate: model-free, fake Generator (no GPU, no exllamav3 import)."""
import os, sys, unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import spec_policy

class Sid:
    def __init__(self, ids): self.ids = list(ids)
    def __len__(self): return len(self.ids)
    def torch(self): return torch.tensor([self.ids], dtype=torch.int32)
    def torch_slice(self, a, b): return torch.tensor([self.ids[a:b]], dtype=torch.int32)
    def truncate(self, n): del self.ids[n:]
    def clear(self): self.ids.clear()

class Seq:
    def __init__(self, sid): self.sequence_ids = sid

class Job:
    def __init__(self, sid): self.sequences = [Seq(sid)]
    def is_prefill_done(self): return True

class Gen:
    num_draft_tokens = 2
    def __init__(self, sid): self.active_jobs = [Job(sid)]; self.draft_calls = 0
    def enqueue(self, job): return None
    def iterate_draftmodel_mtp_gen(self, results):
        self.draft_calls += 1
        return torch.tensor([[0] * self.num_draft_tokens], dtype=torch.long)

def run(rounds, gain_ng, gain_dr, gate="1"):
    """A prompt that repeats a 40-token block, so every round finds a long match. Lookup rounds grow the sequence by
    gain_ng tokens, drafter rounds by gain_dr. Returns (lookup rounds, gated matches)."""
    os.environ["QWSPEC_NGRAM_GATE"] = gate
    block = list(range(100, 140)); sid = Sid(block * 50)
    g = Gen(sid); stats = {}
    spec_policy.install_lookup(g, (2, 7, 16, 3), stats)
    ng = 0
    for _ in range(rounds):
        before = stats.get("ng", 0)
        g.iterate_draftmodel_mtp_gen(None)
        hit = stats.get("ng", 0) > before; ng += hit
        n = len(sid); sid.ids += [100 + (n + i) % 40 for i in range(gain_ng if hit else gain_dr)]
    return ng, stats.get("ng_gated", 0)

class LookupGate(unittest.TestCase):
    def test_good_copies_keep_lookup(self):
        ng, gated = run(100, gain_ng=8, gain_dr=3)
        self.assertEqual(ng, 100); self.assertEqual(gated, 0)

    def test_bad_matches_are_gated_with_probes(self):
        ng, gated = run(200, gain_ng=1, gain_dr=3)
        self.assertLess(ng, 20); self.assertGreater(gated, 150)

    def test_gate_off_restores_old_behaviour(self):
        ng, gated = run(50, gain_ng=1, gain_dr=3, gate="0")
        self.assertEqual(ng, 50); self.assertEqual(gated, 0)

if __name__ == "__main__":
    unittest.main()
