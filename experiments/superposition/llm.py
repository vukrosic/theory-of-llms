"""Superposition in the 8-skill toy LM (toy-suite/sampled.py), Muon-trained.
Trains one model (env vars as in toy-suite, e.g. MOE=8 MSIGN=ns EMB_MULT=8 WIDTH=64), then measures
  - per MLP / expert neuron: how many skills it serves (exp entropy of its mean |activation| per skill)
  - per word: TMS feature dimensionality of its embedding row and output-head row, against word frequency
usage: python3 llm.py SEED [STEPS LR] -> one json line appended to llm_runs.jsonl
"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/my-life/openai-anthropic-goal/muon/toy-suite"))
import torch, torch.nn.functional as F
import sampled, mix, suite

def dimensionality(E):
    n = E.norm(dim=1); Eh = E / n[:, None].clamp_min(1e-9)
    return (n ** 2 / ((Eh @ E.T) ** 2).sum(1).clamp_min(1e-12))

def hidden(layer, h):
    """MLP hidden activations [tokens, neurons] for the input h of the MLP (experts: gate-weighted, zero where not routed)"""
    x = h.reshape(-1, suite.D)
    if not suite.MOE: return F.silu(layer.gate(x)) * layer.up(x)
    m = layer.moe; p = m.router(x).softmax(-1); top = (p + m.bias).topk(suite.MOE_K, -1).indices
    g = torch.zeros_like(p).scatter(1, top, p.gather(1, top)); g = g / g.sum(-1, keepdim=True)
    a = torch.einsum("nd,efd->nef", x, m.gate.weight[:suite.MOE]); a = F.silu(a) * torch.einsum("nd,efd->nef", x, m.up.weight[:suite.MOE])
    return (a * g[:, :, None]).reshape(len(x), -1)

@torch.no_grad()
def analyze(model):
    ins = {}
    for j, l in enumerate(model.layers): l.n2.register_forward_hook(lambda mod, i_, o, j=j: ins.__setitem__(j, o))
    ge = torch.Generator().manual_seed(4242); per = {j: [] for j in range(2)}
    for sk in sampled.SKILLS:
        s, m = sampled.gen(sk, 256, ge); x = s[:, :-1]; model(x); keep = (x != sampled.PAD).reshape(-1)
        for j, l in enumerate(model.layers): per[j].append(hidden(l, ins[j])[keep].abs().mean(0))
    out = {}
    for j in range(2):
        A = torch.stack(per[j], 1)                                   # neurons x skills
        alive = A.max(1).values > 0.05 * A.max()
        P = A[alive] / A[alive].sum(1, keepdim=True); ent = (-(P * P.clamp_min(1e-12).log()).sum(1)).exp()
        out[f"L{j}_skills_per_neuron"] = round(ent.mean().item(), 3); out[f"L{j}_alive"] = int(alive.sum())
        out[f"L{j}_mono"] = round((ent < 2).float().mean().item(), 3)    # share of neurons serving < 2 skills
        tot = A.sum(1); out[f"L{j}_used"] = round((tot.sum() ** 2 / (tot ** 2).sum()).item(), 1)   # participation ratio: neurons effectively in use
        out[f"L{j}_neurons"] = len(tot); out[f"L{j}_top"] = round((tot.max() / tot.median()).item(), 1)   # biggest neuron / typical neuron
    for kind in ("q", "v", "o", "up", "down"):
        er = []
        for l in model.layers:
            Wt = getattr(l.moe, kind).weight[:suite.MOE] if suite.MOE and kind in ("up", "down") else getattr(l, kind).weight[None]
            sv = torch.linalg.svdvals(Wt); er.append(((sv.sum(-1) ** 2 / (sv ** 2).sum(-1)) / sv.shape[-1]).mean().item())
        out[f"erank_{kind}"] = [round(e, 3) for e in er]                  # effective rank / max rank, per layer
        out[f"top_sv_{kind}"] = [round(torch.linalg.svdvals(getattr(l.moe, kind).weight[:suite.MOE] if suite.MOE and kind in ("up", "down") else getattr(l, kind).weight).max().item(), 2) for l in model.layers]
    nv = sampled.NV
    hw = model.head.weight[:nv]
    for name, E in (("emb", model.emb.weight[:nv]), ("head", hw), ("headc", hw - hw.mean(0))):
        d = dimensionality(E)
        out[f"{name}_dim_total"] = round(d.sum().item(), 2)           # = width if every direction is used
        out[f"{name}_dim_by_freq"] = [round(d[a:b].mean().item(), 3) for a, b in ((0, 8), (8, 64), (64, 256), (256, 1024))]
    return out

if __name__ == "__main__":
    seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 260; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.014
    t0 = time.time(); rows = mix.run("whole", seed, steps, lr, every=steps, bs=32); sec = round(time.time() - t0, 1)
    last = rows[-1][1]; wl = sum(w.item() * last[sk][0] for w, sk in zip(sampled.W, sampled.SKILLS))
    cfg = {k: os.environ[k] for k in ("MOE", "WIDTH", "MSIGN", "NS_STEPS", "EMB_MULT", "WD", "TAG") if k in os.environ}
    r = dict(cfg=cfg, seed=seed, steps=steps, lr=lr, sec=sec, wloss=round(wl, 4), acc={sk: last[sk][1] for sk in sampled.SKILLS}, **analyze(mix.last_model))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_runs.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
