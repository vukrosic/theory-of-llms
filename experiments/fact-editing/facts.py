"""Where does the 8-skill toy keep a fact, and can it be edited there?
Rebuilds Geva 2021 (MLP rows as key -> value memories), ROME's causal tracing and rank-one edit (Meng 2022), and
Hase 2023's question (does the traced spot predict where an edit works?) on the default Muon toy.

A fact prompt is [relation, subject, SEP] -> object: relation = the factc or factr skill token, 64 subjects shared by both.
Positions: 0 relation, 1 subject, 2 SEP (the answer is read here).

usage (toy-suite env, e.g. MOE=8 MSIGN=ns EMB_MULT=8):
  python3 facts.py train SEED [LR]   train the default toy (260 steps), save it to $FACT_DIR
  python3 facts.py trace SEED        causal tracing: corrupt the subject, restore one spot, how much of the answer comes back
  python3 facts.py edit SEED         rank-one edit of each fact at each MLP spot: does it work, how many other facts change
"""
import os, sys, json, time
for k, v in (("MOE", "8"), ("MSIGN", "ns"), ("EMB_MULT", "8")): os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../muon/toy-suite"))
import torch, torch.nn.functional as F
import suite, mix, sampled
from sampled import SKILLS, NV, SEP, PAD, L, SUBJ, FACTC, FACTR, batch
from suite import rope, H, D

NS, PACK = int(os.environ.get("NSUBJ", 64)), int(os.environ.get("PACK", 1))   # subjects; facts per training sequence
STOK, PS, ORDER = int(os.environ.get("STOK", 1)), int(os.environ.get("PS", 16)), os.environ.get("ORDER", "rs")
NREL = int(os.environ.get("NREL", 2))
# STOK=2: each subject is 2 tokens (first, last name) from 2 pools of PS words, so no single token names the subject
# ORDER: rs = [relation, subject, SEP] -> object (default toy); sr = [subject, relation] -> object (as in "The Eiffel Tower is in")
# NREL: relations per subject; even ones are asked in factc sequences (common answers), odd ones in factr (rare answers)
CUSTOM = (NS, PACK, STOK, ORDER, NREL) != (64, 1, 1, "rs", 2)
if CUSTOM:
    g_ = torch.Generator().manual_seed(12); w_ = torch.randperm(NV, generator=g_)
    if STOK == 1: SUBJ = w_[:NS, None]
    else:
        pairs = torch.randperm(PS * PS, generator=g_)[:NS]; SUBJ = torch.stack([w_[pairs // PS], w_[PS + pairs % PS]], 1)
    REL = [NV + 2 + SKILLS.index("factc"), NV + 2 + SKILLS.index("factr")] + w_[-(NREL - 2):].tolist()[:max(0, NREL - 2)]
    OBJ = torch.stack([torch.randint(64, (NS,), generator=g_) if r % 2 == 0 else NV // 2 + torch.randint(NV // 2, (NS,), generator=g_) for r in range(NREL)])
    _gen = sampled.gen

    def prompt(r, subj):   # subj: n x STOK -> n x prompt length
        r = r.view(-1, 1).expand(len(subj), 1) if torch.is_tensor(r) else torch.full((len(subj), 1), r)
        return torch.cat([r, subj, torch.full_like(r, SEP)], 1) if ORDER == "rs" else torch.cat([subj, r], 1)

    def gen(skill, n, g):
        if skill not in ("factc", "factr"): return _gen(skill, n, g)
        s = torch.full((n, L + 1), PAD); m = torch.zeros(n, L, dtype=torch.bool); rs = torch.arange(int(skill == "factr"), NREL, 2)
        if ORDER == "sr": s[:, 0] = SEP                               # neutral start token, so the relation is known only at its own position
        o = int(ORDER == "sr")
        for p in range(PACK):
            j = torch.randint(NS, (n,), generator=g); r = rs[torch.randint(len(rs), (n,), generator=g)]
            blk = torch.cat([prompt(torch.tensor(REL)[r], SUBJ[j]), OBJ[r, j][:, None]], 1)
            w = blk.shape[1]; s[:, o + w * p:o + w * (p + 1)] = blk; m[:, o + w * p + w - 2] = True
        return s, m
    sampled.gen = mix.gen = gen
    X = torch.cat([prompt(r, SUBJ) for r in REL])
    if ORDER == "sr": X = torch.cat([torch.full((len(X), 1), SEP), X], 1)
else:
    gen = sampled.gen; SUBJ = SUBJ[:, None]; REL = [NV + 2 + SKILLS.index("factc"), NV + 2 + SKILLS.index("factr")]; OBJ = torch.stack([FACTC, FACTR])
    X = torch.tensor([[r, s.item(), SEP] for r in REL for s in SUBJ])
DIR = os.environ.get("FACT_DIR", "/tmp/fact-editing"); os.makedirs(DIR, exist_ok=True)
TAG = os.environ.get("TAG", ("moe" if suite.MOE else "dense") + (f"_n{NS}p{PACK}t{STOK}{ORDER}r{NREL}" if CUSTOM else ""))
Y = OBJ.flatten(); NF = len(Y)                                          # fact f = relation f // NS, subject f % NS
TP = X.shape[1]; ANS = TP - 1                                           # answer is read at the last prompt position
SPOS = list(range(1, 1 + STOK))                                         # subject positions


def path(seed): return f"{DIR}/{TAG}_s{seed}.pt"


def load(seed):
    torch.manual_seed(0); m = suite.Model(); m.load_state_dict(torch.load(path(seed))); m.eval(); return m


def mlp(l, h, li, hook):
    """MLP / MoE output; hook(li, 'key', k) sees the hidden activations (dense: b,t,F; MoE: E,b*t,F with gates)"""
    if not suite.MOE:
        k = F.silu(l.gate(h)) * l.up(h)
        if hook: k = hook(li, "key", k)
        return l.down(k)
    mo, b, t = l.moe, h.shape[0], h.shape[1]; x = h.reshape(-1, D); p = mo.router(x).softmax(-1)
    top = (p.detach() + mo.bias).topk(suite.MOE_K, -1).indices
    g = torch.zeros_like(p).scatter(1, top, p.gather(1, top)); g = g / g.sum(-1, keepdim=True)
    k = F.silu(torch.einsum("nd,efd->enf", x, mo.gate.weight)) * torch.einsum("nd,efd->enf", x, mo.up.weight)
    if hook: hook(li, "gate", g.view(b, t, -1)); k = hook(li, "key", k)
    return torch.einsum("enf,edf,ne->nd", k, mo.down.weight, g).view(b, t, D)


def fwd(m, x, hook=None):
    """same as m(x) for the default toy, with hook(li, kind, tensor) -> tensor at emb, attn, key, mlp, res"""
    b, t = x.shape; h = m.emb(x)
    if hook: h = hook(-1, "emb", h)
    for li, l in enumerate(m.layers):
        z = l.n1(h)
        q, k, v = (w(z).view(b, t, H, D // H).transpose(1, 2) for w in (l.q, l.k, l.v))
        q, k = rope(l.qn(q)), rope(l.kn(k))
        a = l.o(F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=suite.ATTN_SCALE / (D // H) ** 0.5).transpose(1, 2).reshape(b, t, D))
        if hook: a = hook(li, "attn", a)
        h = h + a
        o = mlp(l, l.n2(h), li, hook)
        if hook: o = hook(li, "mlp", o)
        h = h + o
        if hook: h = hook(li, "res", h)
    return m.head(m.norm(h))


def record(m, x):
    rec = {}
    def hook(li, kind, z): rec[(li, kind)] = z.detach().clone(); return z
    with torch.no_grad(): out = fwd(m, x, hook)
    return out, rec


EV = {}


def skill_losses(m):
    if not EV: EV.update({sk: gen(sk, 128, torch.Generator().manual_seed(999)) for sk in SKILLS})
    out = {}
    with torch.no_grad():
        for sk, (s, msk) in EV.items(): o = m(s[:, :-1]); out[sk] = F.cross_entropy(o[msk], s[:, 1:][msk]).item()
    return out


def train(seed, lr):
    t0 = time.time(); rows = mix.run("whole", seed, 260, lr, every=260, bs=32); m = mix.last_model
    torch.save(m.state_dict(), path(seed))
    with torch.no_grad(): ok = (fwd(m, X)[:, ANS].argmax(-1) == Y).float()
    assert torch.allclose(fwd(m, X), m(X), atol=1e-4)
    print(f"{TAG} seed {seed} lr {lr} {time.time() - t0:.1f}s facts right {ok[:NS].mean():.2f} / {ok[NS:].mean():.2f}  "
          + " ".join(f"{sk}:{rows[-1][1][sk][1]:.2f}" for sk in SKILLS))


SITES = [(li, kind, p) for li in range(2) for kind in ("res", "attn", "mlp") for p in range(TP)]


def trace(seed, n=10, scale=3.0):
    """ROME causal tracing: noise on the subject embedding (scale x embedding std), then put back the clean value at one
    (layer, kind, position); effect = share of the clean answer probability that comes back, per fact"""
    m = load(seed); out, rec = record(m, X)
    pc = out[:, ANS].softmax(-1)[torch.arange(NF), Y]
    xs, ys = X.repeat(n, 1), Y.repeat(n); g = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(xs), D, generator=g) * scale * m.emb.weight.std().item()
    def corrupt(site=None):
        def hook(li, kind, z):
            if kind == "emb": z = z.clone(); z[:, SPOS] += noise[:, None]
            if site and (li, kind) == site[:2]: z = z.clone(); z[:, site[2]] = rec[(li, kind)].repeat(n, 1, 1)[:, site[2]]
            return z
        with torch.no_grad(): return fwd(m, xs, hook)[:, ANS].softmax(-1)[torch.arange(len(xs)), ys].view(n, NF).mean(0)
    p0 = corrupt(); eff = {}
    for s in SITES: eff[s] = ((corrupt(s) - p0) / (pc - p0).clamp_min(1e-6))
    ok = (out[:, ANS].argmax(-1) == Y) & (pc - p0 > 0.2)
    print(f"{TAG} seed {seed}: {ok.sum().item()} facts traced (right, noise drops p by >0.2); clean p {pc[ok].mean():.2f}, corrupted {p0[ok].mean():.2f}")
    for kind in ("res", "attn", "mlp"):
        print(f"  {kind:4s} " + "  ".join(f"L{li} p{p}: {eff[(li, kind, p)][ok].mean():.2f}" for li in range(2) for p in range(TP)))
    xn, _, mn = batch(256, torch.Generator().manual_seed(4243)); _, rn = record(m, xn)
    print("  knockout (output at that spot set to its mean over natural text; share of facts still right):")
    for kind in ("attn", "mlp"):
        mu = {li: rn[(li, kind)][mn].mean(0) for li in range(2)}; row = []
        for li in range(2):
            for p in range(TP):
                def hook(li_, kind_, z):
                    if (li_, kind_) == (li, kind): z = z.clone(); z[:, p] = mu[li]
                    return z
                with torch.no_grad(): row.append(f"L{li} p{p}: {(fwd(m, X, hook)[:, ANS].argmax(-1) == Y)[ok].float().mean():.2f}")
        print(f"  {kind:4s} " + "  ".join(row))
    parts = [("emb", (-1, "emb"))] + [(f"{k}{li}", (li, k)) for li in range(2) for k in ("attn", "mlp")]
    hfin = rec[(1, "res")][:, ANS]; sc = m.norm.weight / hfin.pow(2).mean(-1, keepdim=True).add(m.norm.eps or 1e-6).sqrt()
    U = m.head.weight - m.head.weight.mean(0)                                # answer score = logit minus mean logit
    dla = {nm: ((rec[key][:, ANS] * sc) * U[Y]).sum(-1) for nm, key in parts}
    tot = sum(dla.values())
    print("  answer score from each part (direct, share of total): " + "  ".join(f"{nm}: {(dla[nm][ok] / tot[ok]).mean():.2f}" for nm, _ in parts))
    torch.save(dict(dla={k: v for k, v in dla.items()}, eff={f"{a}_{b}_{c}": v for (a, b, c), v in eff.items()}, ok=ok, pc=pc, p0=p0), f"{DIR}/{TAG}_s{seed}_trace.pt")


def keys_cov(m, li):
    """second moment of the MLP hidden activations over natural text (dense: F x F; MoE: per expert, over its own tokens)"""
    x, _, msk = batch(512, torch.Generator().manual_seed(4242)); _, rec = record(m, x)
    k = rec[(li, "key")]
    if not suite.MOE: kk = k[msk]; return (kk.T @ kk / len(kk))[None]
    use = (rec[(li, "gate")] > 0)[msk]                                      # tokens, E
    C = []
    for e in range(suite.MOE):
        kk = k[e][msk.reshape(-1)][use[:, e]]; C.append(kk.T @ kk / max(1, len(kk)))
    return torch.stack(C)


def edit_one(m, li, pos, f, new, C, rec, lam=1e-2, steps=200):
    """ROME: find the MLP output change d at (li, pos) of prompt f that makes `new` the answer, then write it into the
    down matrix as a rank-one update keyed on that spot's hidden activation (whitened by C). Returns the old weights."""
    d = torch.zeros(D, requires_grad=True); opt = torch.optim.Adam([d], lr=0.05)
    for _ in range(steps):
        def hook(li_, kind, z):
            if (li_, kind) == (li, "mlp"): z = z.clone(); z[:, pos] = z[:, pos] + d
            return z
        lp = fwd(m, X[f:f + 1], hook)[0, ANS].log_softmax(-1)[new]
        if lp.item() > -0.1: break
        opt.zero_grad(); (-lp).backward(); opt.step()
    d = d.detach(); W = (m.layers[li].down if not suite.MOE else m.layers[li].moe.down).weight
    old = W.detach().clone()
    with torch.no_grad():
        Ci = C + lam * C.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(C.shape[-1])
        if not suite.MOE:
            k = rec[(li, "key")][f, pos]; u = torch.linalg.solve(Ci[0], k); W += torch.outer(d, u) / (u @ k)
        else:
            g = rec[(li, "gate")][f, pos]; e = g.argmax().item(); k = rec[(li, "key")][e, f * TP + pos]
            u = torch.linalg.solve(Ci[e], k) if EDIT == "rome" else k; W[e] += torch.outer(d / g[e], u) / (u @ k)
    return old, d.norm().item()


EDIT = os.environ.get("EDIT", "rome")   # rome: key whitened by C^-1 (ROME); raw: key as is, the shape of one Muon step on one fact
FT_LR = float(os.environ.get("FT_LR", 0.005))


def ft_one(m, f, new, which, steps=100):
    """correct one fact by Muon steps on its prompt alone (no momentum) on the chosen matrices, until p(new) > 0.9;
    which: all | mlp0 | mlp1 | attn0 | attn1 | mlp | attn. Returns a copy of the old state and the steps taken"""
    old = {k: v.clone() for k, v in m.state_dict().items()}
    mats = [(getattr(l, n) if not (suite.MOE and n in ("gate", "up", "down")) else getattr(l.moe, n)).weight
            for li, l in enumerate(m.layers) for n in suite.MATS
            if which == "all" or which in (f"mlp{li}", "mlp") and n in ("gate", "up", "down") or which in (f"attn{li}", "attn") and n in "qkvo"]
    for i in range(steps):
        lp = fwd(m, X[f:f + 1])[0, ANS].log_softmax(-1)[new]
        if lp.item() > -0.1: break
        for w in mats: w.grad = None
        (-lp).backward()
        with torch.no_grad():
            for w in mats:
                if w.grad is not None: w -= FT_LR * max(1, w.shape[-2] / w.shape[-1]) ** 0.5 * suite.msign(w.grad)
    return old, i


def edit(seed, nf=int(os.environ.get("NEDIT", 32))):
    m = load(seed); out, rec = record(m, X); pred0 = out[:, ANS].argmax(-1); base = skill_losses(m)
    right = (pred0 == Y); g = torch.Generator().manual_seed(seed + 77); res = []
    Cs = {li: keys_cov(m, li) for li in range(2)}
    facts = [f for f in torch.randperm(NF, generator=torch.Generator().manual_seed(5)).tolist() if right[f]][:nf]
    sites = [tuple(map(int, x.split(":"))) for x in os.environ["ESITES"].split(",")] if "ESITES" in os.environ and EDIT != "ft" else \
        [(0, p) for p in range(1, TP)] + [(1, ANS)] if EDIT != "ft" else os.environ.get("FT_ON", "all,mlp1,attn1,mlp,attn").split(",")
    for site in sites:
        li, pos = site if EDIT != "ft" else (site, -1)
        t0 = time.time()
        if True:
            for f in facts:
                pool = OBJ[f // NS]
                new = pool[torch.randint(NS, (1,), generator=g)].item()
                while new == Y[f].item(): new = pool[torch.randint(NS, (1,), generator=g)].item()
                if EDIT == "ft": old, dn = ft_one(m, f, new, li)
                else: old, dn = edit_one(m, li, pos, f, new, Cs[li], rec)
                with torch.no_grad(): pr = fwd(m, X)[:, ANS].argmax(-1)
                ch = (pr != pred0); ch[f] = False
                same_rel = torch.arange(NF) // NS == f // NS; same_subj = (torch.arange(NF) % NS == f % NS) & ~same_rel
                sl = skill_losses(m)
                if EDIT == "ft": m.load_state_dict(old)
                else:
                    W = (m.layers[li].down if not suite.MOE else m.layers[li].moe.down).weight
                    with torch.no_grad(): W.copy_(old)
                res.append(dict(li=li, pos=pos, f=f, ok=int(pr[f].item() == new), dn=dn, changed=int(ch.sum()), sib=(ch & same_subj).float().sum().item() / max(1, NREL - 1),
                                same_rel=int((ch & same_rel).sum()), other_rel=int((ch & ~same_rel & ~same_subj).sum()),
                                dloss={sk: round(sl[sk] - base[sk], 4) for sk in SKILLS}))
            r = [x for x in res if (x["li"], x["pos"]) == (li, pos)]
            mean = lambda k: sum(x[k] for x in r) / len(r)
            other_sk = sum(sum(v for sk, v in x["dloss"].items() if not sk.startswith("fact")) for x in r) / len(r) / 6
            print(f"{TAG} {EDIT} seed {seed} {f'L{li} pos{pos}' if EDIT != 'ft' else li}: edit works {mean('ok'):.2f}  other facts changed {mean('changed'):.1f}/{NF - 1} "
                  f"(same subject {mean('sib'):.2f} of its other facts, same relation {mean('same_rel'):.1f}, rest {mean('other_rel'):.1f})  "
                  f"other skills loss +{other_sk:.4f}  |d| {mean('dn'):.2f}  {time.time() - t0:.0f}s")
    with open(f"{DIR}/{TAG}_s{seed}_edit_{EDIT}.json", "w") as fh: json.dump(res, fh)


if __name__ == "__main__":
    cmd, seed = sys.argv[1], int(sys.argv[2])
    if cmd == "train": train(seed, float(sys.argv[3]) if len(sys.argv) > 3 else 0.014)
    elif cmd == "trace": trace(seed)
    elif cmd == "edit": edit(seed)
