"""Pack-side GPU measurements, run with the backend python (torch + exllamav3 on PYTHONPATH). CLI only; stages parse its JSON.
  gpu_eval.py parity PACK DOCS.pt REFDIR OUT.json            KL / top-1 / dNLL per doc (fp16 KV) against the streamed reference logits
  gpu_eval.py kv     PACK DOCS.pt REFDIR OUT.json B1 B2 ...  tail KL of symmetric quantized KV (k<b>v<b>) at the long doc, vs reference and vs k8v8
KL maths and the real-quantizer forward are adapted from nanb kv_kl.py (drill 0)."""
import gc
import json
import sys

import torch


def stats(a, b, tgt):
    """a = reference logits, b = test logits, both [T,V]; tgt = next tokens [T]. -> KL(a||b), top-1 agreement, dNLL."""
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    kl = (la.exp() * (la - lb)).sum(-1).mean().item()
    t1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    dn = (-lb.gather(1, tgt[:, None]).mean() + la.gather(1, tgt[:, None]).mean()).item()
    return kl, t1, dn


def load(pack, T, kv=None):
    from exllamav3 import Cache, CacheLayer_quant, Config, Model
    m = Model.from_config(Config.from_directory(pack))
    cache = Cache(m, max_num_tokens=T + 256, layer_type=CacheLayer_quant, k_bits=kv[0], v_bits=kv[1]) if kv else Cache(m, max_num_tokens=T + 256)
    m.load(progressbar=False)
    return m, cache


def forward_tail(m, cache, ids, tail):
    T = len(ids)
    x = torch.tensor([ids])
    bt = torch.arange((T + 256) // 256, dtype=torch.int32)[None]
    with torch.inference_mode():
        for a in range(0, T - tail, 2048):
            e = min(a + 2048, T - tail)
            m.prefill(x[:, a:e], {"attn_mode": "flash_attn", "cache": cache, "block_table": bt, "cache_seqlens": torch.tensor([a], dtype=torch.int32)})
        lg = m.forward(x[:, T - tail:], {"attn_mode": "flash_attn", "cache": cache, "block_table": bt, "cache_seqlens": torch.tensor([T - tail], dtype=torch.int32), "last_tokens_only": tail})
    return lg[0].float()


def free(m, cache):
    m.unload(); del cache, m; gc.collect(); torch.cuda.empty_cache()


def parity(pack, docs, ref, out):
    d = torch.load(docs)
    res = {}
    m, cache = load(pack, max(len(i) for i in d["ids"]))
    for n, ids in zip(d["names"], d["ids"]):
        r = torch.load(f"{ref}/logits_{n}.pt").float().cuda()       # [T,V]; row i predicts ids[i+1]
        T = len(ids)
        lg = forward_tail(m, cache, ids, T)
        kl, t1, dn = stats(r[:-1], lg[:-1], torch.tensor(ids[1:]).cuda())
        res[n] = dict(kl=kl, top1=t1, dnll=dn, tokens=T - 1)
        print(n, res[n], flush=True)
    free(m, cache)
    json.dump(res, open(out, "w"), indent=1)


def kv(pack, docs, ref, out, bits):
    d = torch.load(docs)
    ids = d["ids"][0]
    r = torch.load(f"{ref}/logits_{d['names'][0]}.pt").float().cuda()
    tail = r.shape[0]
    tgt = torch.tensor(ids[-tail + 1:]).cuda(); r = r[:-1]
    res, lgs = {}, {}
    for b in [8] + [x for x in bits if x != 8]:
        m, cache = load(pack, len(ids), (b, b))
        lg = forward_tail(m, cache, ids, tail)[:-1]
        free(m, cache)
        kl, t1, dn = stats(r, lg, tgt)
        res[b] = dict(kl_ref=kl, top1_ref=t1, dnll=dn)
        lgs[b] = lg.half()
        if b != 8:
            k2, t2, _ = stats(lgs[8], lg, tgt)
            res[b].update(kl_vs_k8=k2, top1_vs_k8=t2)
        print(f"k{b}v{b}", res[b], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        parity(*sys.argv[2:6])
    elif cmd == "kv":
        kv(*sys.argv[2:6], [int(x) for x in sys.argv[6:]])
