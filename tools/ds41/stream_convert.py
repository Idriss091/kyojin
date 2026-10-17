"""Layer-wise streaming converter: remote FP4/FP8 checkpoint -> EXL3.

    python tools/ds41/stream_convert.py --map tensor_map.json --out OUT --target 2.3 \
        [--freq freq.json] [--layers 0-39] [--include global,mtp,engram] [--device cuda:0] [--dry-run]

Nothing is downloaded as a whole shard. Per unit (global, one decoder layer, one MTP layer, one
engram table) the converter reads each tensor with Range requests, dequantizes it to bf16 and hands
every linear to exllamav3's `quantize_exl3` (no calibration Hessian: the uncalibrated path, as used
for modules without calibration data). Each unit becomes one output shard `<unit>.safetensors`,
published by atomic rename together with a `<unit>.done.json` marker; a rerun skips finished units.
When every unit is done, `model.safetensors.index.json`, `config.json` (with `quantization_config`)
and `quantization_config.json` are written.

Peak RAM (host), per unit: the unit's quantized output is held until its shard is written, at most
about 5.2 GB (a routed layer with every expert at 3 bits), plus about 3 GB of working tensors
(largest: the output head in fp32, 2.6 GB) plus `--prefetch` in-flight fetches (a few MB each).
Plan on 10 GB. Engram tables never enter RAM: they are copied byte for byte in 64 MiB chunks
straight into the output file. GPU memory: the quantizer working set of one linear.

--dry-run does everything except the quantization (and the engram byte copy): it fetches and
dequantizes each tensor, then records the bf16 size, the planned output size and the plan in
`<out>/dry_run/`. It needs no GPU and no exllamav3 import.
"""

import argparse
import gc
import json
import os
import resource
import struct
import sys
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dequant as D  # noqa: E402
import fetch as F  # noqa: E402
import plan as P  # noqa: E402

SAFE_DTYPE = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
}


# ----------------------------------------------------------------------------------------------
# Output side
# ----------------------------------------------------------------------------------------------

def write_streamed(path: str, items: list) -> None:
    """Write a safetensors file from [(name, dtype_str, shape, nbytes, chunk_iterable)] without
    holding the data in memory. The header is built from the declared sizes."""
    header, pos = OrderedDict(), 0
    for name, dt, shape, nbytes, _ in items:
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [pos, pos + nbytes]}
        pos += nbytes
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, _, _, nbytes, chunks in items:
            written = 0
            for c in chunks:
                f.write(c)
                written += len(c)
            if written != nbytes:
                raise IOError(f"{name}: wrote {written} of {nbytes} bytes")


def save_tensors(path: str, tensors: dict) -> None:
    from safetensors.torch import save_file
    save_file(tensors, path, metadata={"format": "pt"})


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------
# Quantizer hand-off
# ----------------------------------------------------------------------------------------------

class Exl3Quantizer:
    """Thin wrapper around exllamav3.modules.quant.exl3_lib.quantize.quantize_exl3."""

    def __init__(self, device: str, codebook: str = "mul1", out_scales: bool = True):
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
        self._quantize = quantize_exl3
        self.device = device
        self.codebook = codebook
        self.out_scales = out_scales

    def __call__(self, key: str, w_bf16: torch.Tensor, bits: int) -> dict:
        """w_bf16: [out, in]. Returns {suffix: cpu tensor} for the EXL3 tensors of this linear."""
        out_f, in_f = w_bf16.shape
        H_data = {
            "H": torch.empty((in_f, in_f), device="meta"),  # meta Hessian = uncalibrated path
            "first_key": key, "count": 0, "finalized": False, "num_total": 0,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=self.device),
            "device": self.device,
        }
        qargs = {"seed": zlib.crc32(key.encode()), "K": bits, "devices": [self.device],
                 "device_ratios": None, "apply_out_scales": self.out_scales}
        if self.codebook in ("mul1", "mcg"):
            qargs[self.codebook] = True
        weight = w_bf16.float().T.contiguous()
        _, _, out = self._quantize(weight, H_data, qargs, False)
        return {k: v.cpu().contiguous() for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# Unit processing
# ----------------------------------------------------------------------------------------------

def group_jobs(entries: list) -> list:
    """Group plan entries of a unit by source weight so a shared tensor is fetched once."""
    jobs = OrderedDict()
    for e in entries:
        k = e.get("weight") or e["name"]
        jobs.setdefault(k, []).append(e)
    return list(jobs.values())


def fetch_job(fetcher: F.TensorFetcher, job: list):
    e = job[0]
    if e["action"] == "quant":
        w = fetcher.fetch(e["weight"])
        s = fetcher.fetch(e["scale"]) if e.get("scale") else None
        return w, s
    return fetcher.fetch(e["name"]), None


def slice_rows(w, s, rows):
    if rows is None:
        return w, s
    a, b = rows
    bo = w.shape[0] // s.shape[0] if s is not None else 1
    return w[a:b], (s[a // bo:b // bo] if s is not None else None)


def process_unit(unit, entries, fetcher, out_dir, dry, quantizer, prefetch, log):
    t0 = time.time()
    tensors = {}
    stored = {}   # module key -> record for quantization_config
    stats = {"bf16_bytes": 0, "planned_bytes": 0, "fetched_bytes": 0, "tensors": 0}
    jobs = group_jobs(entries)
    final = os.path.join(out_dir, f"{unit}.safetensors")

    if jobs and jobs[0][0]["action"] == "copy_raw":
        # Engram: raw FP8 + scale bytes, streamed straight to disk
        items = []
        for e in sorted(entries, key=lambda e: e["name"]):
            info = fetcher.info(e["name"])
            stats["planned_bytes"] += info["length"]
            stats["tensors"] += 1
            if not dry:
                items.append((e["name"], info["dtype"], info["shape"], info["length"],
                              fetcher.iter_chunks(e["name"])))
                stored[e["name"]] = {"dtype": info["dtype"], "shape": info["shape"]}
        if not dry:
            write_streamed(final + ".tmp", items)
            stats["fetched_bytes"] = stats["planned_bytes"]
    else:
        with ThreadPoolExecutor(max_workers=max(1, prefetch)) as ex:
            futs = [ex.submit(fetch_job, fetcher, j) for j in jobs[:prefetch]]
            nxt = len(futs)
            for i, job in enumerate(jobs):
                w, s = futs[i].result()
                futs[i] = None
                if nxt < len(jobs):
                    futs.append(ex.submit(fetch_job, fetcher, jobs[nxt]))
                    nxt += 1
                else:
                    futs.append(None)
                e0 = job[0]
                stats["fetched_bytes"] += w.numel() * w.element_size() + (s.numel() if s is not None else 0)
                for e in job:
                    stats["tensors"] += 1
                    if e["action"] == "copy":
                        stats["bf16_bytes"] += w.numel() * w.element_size()
                        stats["planned_bytes"] += e["out_bytes"]
                        if not dry:
                            tensors[e["name"]] = w
                            stored[e["name"]] = {"dtype": fetcher.info(e["name"])["dtype"], "shape": list(w.shape)}
                        continue
                    ws, ss = slice_rows(w, s, e["rows"])
                    bf = D.dequant(ws, ss)
                    assert tuple(bf.shape) == (e["out"], e["in"]), (e["key"], bf.shape, e["out"], e["in"])
                    stats["bf16_bytes"] += bf.numel() * 2
                    stats["planned_bytes"] += e["out_bytes"]
                    if not dry:
                        q = quantizer(e["key"], bf, e["bits"])
                        rec = {"bits": e["bits"], "stored": {}}
                        for suffix, t in q.items():
                            tensors[f"{e['key']}.{suffix}"] = t
                            rec["stored"][suffix] = {"dtype": SAFE_DTYPE[t.dtype], "shape": list(t.shape)}
                            if suffix == "mul1":
                                rec["mul1_multiplier"] = int(t.item()) & 0xFFFFFFFF
                        stored[e["key"]] = rec
                    del bf, ws, ss
                del w, s
        if not dry:
            save_tensors(final + ".tmp", tensors)
    if not dry:
        os.replace(final + ".tmp", final)
        marker = {"unit": unit, "file": f"{unit}.safetensors", "size": os.path.getsize(final), "stored": stored,
                  **stats}
        C.atomic_write_json(os.path.join(out_dir, f"{unit}.done.json"), marker)
    stats.update({"seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)})
    del tensors
    free_memory()
    log(f"{unit}: {stats}")
    return stats


def unit_done(unit, out_dir, dry) -> bool:
    if dry:
        return os.path.exists(os.path.join(out_dir, "dry_run", f"{unit}.json"))
    mp = os.path.join(out_dir, f"{unit}.done.json")
    if not os.path.exists(mp):
        return False
    m = json.load(open(mp))
    fp = os.path.join(out_dir, m["file"])
    return os.path.exists(fp) and os.path.getsize(fp) == m["size"]


def select_units(all_units, layers, include):
    sel = []
    lo, hi = (None, None)
    if layers:
        a, _, b = layers.partition("-")
        lo, hi = int(a), int(b or a)
    inc = set(include.split(",")) if include else set()
    for u in all_units:
        if u.endswith(".engram"):
            ok = "engram" in inc or (layers is None and not include)
        elif u == "global":
            ok = "global" in inc or (layers is None and not include)
        elif u.startswith("mtp."):
            ok = "mtp" in inc or (layers is None and not include)
        else:
            n = int(u.split(".")[1])
            ok = (lo is None or lo <= n <= hi) if (layers or not include) else False
        if ok:
            sel.append(u)
    return sel


def finalize(plan, out_dir, tmap, log):
    """Index, config.json and quantization_config.json from the unit markers."""
    weight_map, storage = {}, {}
    total = 0
    for u in P.units(plan):
        m = json.load(open(os.path.join(out_dir, f"{u}.done.json")))
        total += m["size"]
        for key, rec in m["stored"].items():
            if "bits" in rec:
                for suffix in rec["stored"]:
                    weight_map[f"{key}.{suffix}"] = m["file"]
                storage[key] = {"quant_format": "exl3", "bits_per_weight": rec["bits"],
                                "stored_tensors": {f"{key}.{s}": v for s, v in rec["stored"].items()},
                                **({"mul1_multiplier": rec["mul1_multiplier"]} if "mul1_multiplier" in rec else {})}
            else:
                weight_map[key] = m["file"]
                storage[key] = {"stored_tensors": {key: rec}}
    C.atomic_write_json(os.path.join(out_dir, "model.safetensors.index.json"),
                        {"metadata": {"total_size": total}, "weight_map": weight_map})
    cfg = C.fetch_json("config.json", tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION))
    s = plan["summary"]
    bits = plan["bits"]
    qcfg = {"quant_method": "exl3", "bits": round(s["routed_avg_bpw"], 4), "head_bits": bits["head"],
            "mtp_bits": bits["mtp"], "codebook": "mul1", "out_scales": "always",
            "plan": {"target": plan["target"], "bits": bits, "expert_frequency": plan["freq"]},
            "original_quantization_config": cfg.get("quantization_config")}
    cfg["quantization_config"] = qcfg
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    with open(os.path.join(out_dir, "quantization_config.json"), "w") as f:
        json.dump({**qcfg, "tensor_storage": storage}, f, indent=4)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        try:
            data = C.range_get(C.file_url(name, tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION)), 0, 0,
                               with_total=True)[1]
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(C.range_get(C.file_url(name, tmap.get("repo", C.REPO), tmap.get("revision", C.REVISION)),
                                    0, data - 1))
        except Exception as e:  # optional files
            log(f"skip {name}: {e}")
    log(f"finalized: {len(weight_map)} tensors, {total / 1e9:.1f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=2.3)
    ap.add_argument("--freq")
    ap.add_argument("--plan", help="use a saved plan.json instead of building one")
    ap.add_argument("--layers", help="decoder layers only, e.g. 0-1")
    ap.add_argument("--include", help="extra units: global,mtp,engram")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--codebook", default="mul1", choices=["mul1", "mcg"])
    ap.add_argument("--prefetch", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-experts", type=int, help="testing aid: keep only routed experts below this index")
    for k, v in P.DEFAULT_BITS.items():
        ap.add_argument(f"--{k}-bits", type=int, default=v)
    a = ap.parse_args(argv)

    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    tmap = json.load(open(a.map))
    os.makedirs(a.out, exist_ok=True)
    if a.plan:
        plan = json.load(open(a.plan))
    else:
        plan = P.build_plan(tmap, a.target, P.load_freq(a.freq) if a.freq else None,
                            {k: getattr(a, f"{k}_bits") for k in P.DEFAULT_BITS})
    if a.max_experts is not None:
        plan["entries"] = [e for e in plan["entries"]
                           if e["kind"] != "routed" or int(e["key"].split(".")[4]) < a.max_experts]
        plan["summary"] = P.summarize(plan["entries"])
    C.atomic_write_json(os.path.join(a.out, "plan.json"), plan)
    all_units = P.units(plan)
    sel = select_units(all_units, a.layers, a.include)
    log(f"{len(sel)}/{len(all_units)} units selected, plan {plan['summary']['total_gb']:.1f} GB, "
        f"routed avg {plan['summary']['routed_avg_bpw']:.3f} bpw, dry_run={a.dry_run}")

    by_unit = {}
    for e in plan["entries"]:
        by_unit.setdefault(e["unit"], []).append(e)
    fetcher = F.TensorFetcher(tmap)
    quantizer = None if a.dry_run else Exl3Quantizer(a.device, a.codebook)
    if a.dry_run:
        os.makedirs(os.path.join(a.out, "dry_run"), exist_ok=True)

    for u in sel:
        if unit_done(u, a.out, a.dry_run):
            log(f"{u}: done, skipping")
            continue
        st = process_unit(u, by_unit[u], fetcher, a.out, a.dry_run, quantizer, a.prefetch, log)
        if a.dry_run:
            C.atomic_write_json(os.path.join(a.out, "dry_run", f"{u}.json"), st)

    if a.dry_run:
        rep = {u: json.load(open(os.path.join(a.out, "dry_run", f"{u}.json")))
               for u in all_units if os.path.exists(os.path.join(a.out, "dry_run", f"{u}.json"))}
        C.atomic_write_json(os.path.join(a.out, "dry_run_report.json"),
                            {"units": rep,
                             "bf16_bytes": sum(r["bf16_bytes"] for r in rep.values()),
                             "planned_bytes": sum(r["planned_bytes"] for r in rep.values())})
        log("dry run report written")
    elif all(unit_done(u, a.out, False) for u in all_units):
        finalize(plan, a.out, tmap, log)
    else:
        log("not all units done; finalize skipped")


if __name__ == "__main__":
    main()
