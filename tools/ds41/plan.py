"""Bits-per-weight plan for the streaming conversion.

    python tools/ds41/plan.py --map tensor_map.json --target 2.3 [--freq freq.json] --out plan.json

Every tensor of the checkpoint is assigned one action:

* quant      linear weight quantized to EXL3 at an integer bit width K (the format stores K bits per
             weight plus per-row/column sign vectors)
* copy       small tensor stored as it is (norms, router gate, hyper-connection params, embeddings,
             vision tower, bf16 projections)
* copy_raw   engram tables (and the engram projection), raw FP8 + ue8m0 scale bytes, never quantized

Defaults: attention and indexer 5 bits, shared experts 5, MTP block (attention, shared and routed
experts, input projection) 4, output head 6, routed experts in two tiers (hot 3 bits, cold 2 bits).
EXL3 bit widths are integers per tensor, so the routed average is reached by mixing tiers: the hot
fraction is (target - cold) / (hot - cold). With an expert-frequency file ({layer: {expert: count}})
the hot tier goes to the most used experts of each layer; without it the hot experts are spread
evenly over the expert index.

The target applies to the backbone routed experts; the plan summary lists every group separately.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DEFAULT_BITS = {"attn": 5, "shared": 5, "mtp": 4, "head": 6, "hot": 3, "cold": 2}
N_EXPERTS = 384
O_GROUPS = 8

RE_EXPERT = re.compile(r"^(layers\.\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
RE_LAYER = re.compile(r"^(layers|mtp)\.(\d+)\.")
RE_ENGRAM = re.compile(r"^layers\.(\d+)\.engram\.")


def exl3_bytes(in_f: int, out_f: int, k: int) -> int:
    """Size of the EXL3 tensors of one linear: trellis int16 [in/16, out/16, 16k], suh, svh, mul1."""
    return (in_f // 16) * (out_f // 16) * 16 * k * 2 + in_f * 2 + out_f * 2 + 4


def unit_of(name: str) -> str:
    m = RE_ENGRAM.match(name)
    if m:
        return f"layers.{m.group(1)}.engram"
    m = RE_LAYER.match(name)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return "global"


def load_freq(path):
    """Expert-frequency JSON -> {layer_int: {expert_int: count}}. Accepts '3' or 'layers.3' keys."""
    raw = json.load(open(path))
    out = {}
    for lk, d in raw.items():
        li = int(str(lk).split(".")[-1])
        out[li] = {int(e): float(c) for e, c in d.items()}
    return out


def hot_experts(layer: int, n_hot: int, freq) -> set:
    """Pick the hot expert set of one layer."""
    if n_hot <= 0:
        return set()
    if n_hot >= N_EXPERTS:
        return set(range(N_EXPERTS))
    if freq and layer in freq:
        counts = freq[layer]
        order = sorted(range(N_EXPERTS), key=lambda e: (-counts.get(e, 0.0), e))
        return set(order[:n_hot])
    # evenly spread: expert e is hot when floor((e+1) * n_hot / N) increases
    return {e for e in range(N_EXPERTS) if ((e + 1) * n_hot) // N_EXPERTS > (e * n_hot) // N_EXPERTS}


def build_plan(tmap: dict, target: float, freq=None, bits=None) -> dict:
    b = dict(DEFAULT_BITS)
    b.update(bits or {})
    hot, cold = b["hot"], b["cold"]
    frac = 0.0 if hot == cold else min(1.0, max(0.0, (target - cold) / (hot - cold)))
    n_hot = round(frac * N_EXPERTS)
    T = tmap["tensors"]
    hot_sets = {}
    entries = []
    handled = set()

    def add_quant(key, kind, wname, sname, rows, in_f, out_f, k, tier=None):
        entries.append({
            "action": "quant", "key": key, "kind": kind, "unit": unit_of(wname),
            "weight": wname, "scale": sname, "rows": rows, "in": in_f, "out": out_f,
            "bits": k, "tier": tier, "src_bytes": T[wname]["length"] * (out_f / T[wname]["shape"][0]),
            "out_bytes": exl3_bytes(in_f, out_f, k),
        })

    for name in sorted(T):
        if name in handled:
            continue
        t = T[name]
        unit = unit_of(name)
        if RE_ENGRAM.match(name):
            entries.append({"action": "copy_raw", "name": name, "unit": unit, "kind": "engram",
                            "out_bytes": t["length"], "src_bytes": t["length"]})
            continue
        if name.endswith(".scale") and name[:-6] + ".weight" in T:
            continue  # consumed with its weight
        if name.endswith(".weight") and name[:-7] + ".scale" in T:
            sname = name[:-7] + ".scale"
            handled.add(sname)
            out_f = t["shape"][0]
            in_f = t["shape"][1] * (2 if t["dtype"] == "I8" else 1)
            base = name[:-7]
            m = RE_EXPERT.match(name)
            if m:
                layer, e = int(m.group(1).split(".")[1]), int(m.group(2))
                if layer not in hot_sets:
                    hot_sets[layer] = hot_experts(layer, n_hot, freq)
                is_hot = e in hot_sets[layer]
                add_quant(base, "routed", name, sname, None, in_f, out_f, hot if is_hot else cold,
                          "hot" if is_hot else "cold")
            elif name.startswith("mtp."):
                add_quant(base, "mtp", name, sname, None, in_f, out_f, b["mtp"])
            elif ".shared_experts." in name:
                add_quant(base, "shared", name, sname, None, in_f, out_f, b["shared"])
            elif name.endswith("attn.wo_a.weight"):
                g = out_f // O_GROUPS
                for i in range(O_GROUPS):
                    add_quant(f"{base}.slice.{i}", "attn", name, sname, [i * g, (i + 1) * g], in_f, g, b["attn"])
            else:
                add_quant(base, "attn", name, sname, None, in_f, out_f, b["attn"])
            continue
        if name == "head.weight":
            add_quant("head", "head", name, None, None, t["shape"][1], t["shape"][0], b["head"])
            continue
        entries.append({"action": "copy", "name": name, "unit": unit, "kind": "copy",
                        "out_bytes": t["length"], "src_bytes": t["length"]})

    return {"target": target, "bits": b, "hot_fraction": frac, "experts_hot_per_layer": n_hot,
            "freq": bool(freq), "entries": entries, "summary": summarize(entries)}


def summarize(entries) -> dict:
    groups = {}
    params = {}
    for e in entries:
        g = e["kind"]
        if e["action"] == "quant":
            g = f"{g}:{e['bits']}b"
            params[g] = params.get(g, 0) + e["in"] * e["out"]
        groups[g] = groups.get(g, 0) + e["out_bytes"]
    total = sum(groups.values())
    routed_w = sum(e["in"] * e["out"] for e in entries if e["kind"] == "routed")
    routed_bits = sum(e["in"] * e["out"] * e["bits"] for e in entries if e["kind"] == "routed")
    return {
        "total_bytes": total, "total_gb": total / 1e9,
        "groups_gb": {k: v / 1e9 for k, v in sorted(groups.items())},
        "routed_avg_bpw": routed_bits / routed_w if routed_w else None,
        "quant_params": params,
    }


def units(plan) -> list:
    """Unit ids in processing order (global, layers, mtp, engram tables last)."""
    seen = []
    for e in plan["entries"]:
        if e["unit"] not in seen:
            seen.append(e["unit"])

    def order(u):
        if u == "global":
            return (0, 0, "")
        if u.endswith(".engram"):
            return (3, int(u.split(".")[1]), "")
        pre, idx = u.split(".")
        return (1 if pre == "layers" else 2, int(idx), "")
    return sorted(seen, key=order)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--target", type=float, default=2.3, help="average routed-expert bpw (2.0/2.3/2.6)")
    ap.add_argument("--freq", help="expert frequency JSON: layer -> expert -> count")
    ap.add_argument("--out")
    for k, v in DEFAULT_BITS.items():
        ap.add_argument(f"--{k}-bits", type=int, default=v)
    a = ap.parse_args()
    tm = json.load(open(a.map))
    p = build_plan(tm, a.target, load_freq(a.freq) if a.freq else None,
                   {k: getattr(a, f"{k}_bits") for k in DEFAULT_BITS})
    if a.out:
        C.atomic_write_json(a.out, p)
    s = p["summary"]
    print(f"target {a.target}: hot fraction {p['hot_fraction']:.3f}, routed avg {s['routed_avg_bpw']:.4f} bpw, "
          f"total {s['total_gb']:.1f} GB")
    for k, v in s["groups_gb"].items():
        print(f"  {k:14s} {v:9.2f} GB")
