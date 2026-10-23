"""Can Muon's best lr be read off the data before training?
usage: python3 lr.py gen          -> data/NAME.pt for every dataset + measures.json (data measures, no training)
       python3 lr.py sweep W      -> runs.jsonl: final eval loss per (dataset, Muon lr, seed), W worker processes
       python3 lr.py fit          -> best lr per dataset + how well each data measure tracks it
data: Markov chain over V tokens; knobs s (word-frequency skew), beta (next-word predictability), p (share of random
junk tokens), d (share of sequences from a 64-sequence repeat pool), c (share of sequences whose 2nd half copies the 1st)
model: 2 layers, D=64, 4 heads, RoPE, QK-norm, SwiGLU, RMSNorm, no biases; Muon (5 Newton-Schulz steps, momentum 0.95
Nesterov, step x max(1, rows/cols)^0.5) on block matrices, fixed-lr Adam-style step on embedding, head, norm gains
"""
import sys, os, json, gzip, math, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
V, T, B, H, NL = 256, int(os.environ.get("T", 32)), int(os.environ.get("B", 32)), 4, int(os.environ.get("NL", 2))   # knobs: batch B (default 32), depth NL (default 2)
D = int(os.environ.get("D", 64)); FF = 176 * D // 64                     # width knob (default 64, FF 176)
STEPS = int(os.environ.get("STEPS", 150)); NTR, NEV = STEPS * B, 512
SUF = "" if STEPS == 150 else str(STEPS)                          # other lengths: data600/, runs600.jsonl
OTHER_LR = float(os.environ.get("OTHER_LR", 0.02))
LRS = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 0.057, 0.08]
HERE = os.path.dirname(os.path.abspath(__file__))


def datasets():
    base = dict(s=1.0, beta=2.0, p=0.0, d=0.0, c=0.0); out = {"base": base}
    for k, vals in dict(s=[0, 0.5, 1.5, 2], beta=[0.5, 1, 4, 8], p=[0.1, 0.3, 0.5], d=[0.25, 0.5, 0.9], c=[0.5, 1.0]).items():
        for v in vals: out[f"{k}{v}"] = {**base, k: v}
    g = torch.Generator().manual_seed(123)
    for i in range(10):
        r = torch.rand(5, generator=g).tolist()
        out[f"mix{i}"] = dict(s=2 * r[0], beta=0.5 * 16 ** r[1], p=0.5 * r[2], d=0.9 * r[3] ** 2, c=r[4])
    return out


def make(prm, n, seed):
    uni = (torch.arange(V) + 1.0) ** (-prm["s"]); uni = uni / uni.sum()
    P = (prm["beta"] * torch.randn(V, V, generator=torch.Generator().manual_seed(7)) + uni.log()).softmax(-1)   # same table for train and eval

    def chain(m, g):
        x = torch.empty(m, T + 1, dtype=torch.long); x[:, 0] = torch.multinomial(uni, m, replacement=True, generator=g)
        for t in range(T): x[:, t + 1] = torch.multinomial(P[x[:, t]], 1, generator=g)[:, 0]
        cp = torch.rand(m, generator=g) < prm["c"]; x[cp, T // 2 + 1:] = x[cp, 1:T // 2 + 1]
        return x
    g = torch.Generator().manual_seed(seed); x = chain(n, g)
    pool = chain(64, torch.Generator().manual_seed(11))
    dp = torch.rand(n, generator=g) < prm["d"]; x[dp] = pool[torch.randint(64, (int(dp.sum()),), generator=g)]
    junk = torch.rand(x.shape, generator=g) < prm["p"]; x[junk] = torch.randint(V, (int(junk.sum()),), generator=g)
    return x


class RMS(nn.Module):
    def __init__(s, n): super().__init__(); s.w = nn.Parameter(torch.ones(n))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w


_f = 1.0 / 10000 ** (torch.arange(0, D // H, 2) / (D // H)); _a = torch.arange(T)[:, None] * _f[None]
COS, SIN = _a.cos(), _a.sin()
def rope(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]; c, s = COS[:x.shape[-2]], SIN[:x.shape[-2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s):
        super().__init__(); L = lambda i, o: nn.Linear(i, o, bias=False)
        s.n1, s.n2, s.qn, s.kn = RMS(D), RMS(D), RMS(D // H), RMS(D // H)
        s.q, s.k, s.v, s.o, s.gate, s.up, s.down = L(D, D), L(D, D), L(D, D), L(D, D), L(D, FF), L(D, FF), L(FF, D)
    def forward(s, x):
        b, t, _ = x.shape; h = s.n1(x); sp = lambda z: z.view(b, t, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(rope(s.qn(sp(s.q(h)))), rope(s.kn(sp(s.k(h)))), sp(s.v(h)), is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(b, t, D)); h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.blocks = nn.ModuleList(Block() for _ in range(NL))
        s.norm = RMS(D); s.head = nn.Linear(D, V, bias=False)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return s.head(s.norm(h))
    def loss(s, x): return F.cross_entropy(s(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))


def ns(X):                                                              # batched: X is (n, rows <= cols)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5): A = X @ X.mT; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X


def split(m):
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and n.startswith("blocks")]
    ids = {id(p) for p in mats}; return mats, [p for p in m.parameters() if id(p) not in ids]


def train(data, lr, seed):
    tr, ev = data["train"], data["eval"]; torch.manual_seed(seed); m = LM(); mats, rest = split(m)
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))
    for i in range(STEPS):
        f = min(1.0, (i + 1) / 20) * min(1.0, (1 - i / STEPS) / 0.5)
        loss = m.loss(tr[order[i * B:(i + 1) * B]]); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            u = []
            for p, b in zip(mats, buf): b.mul_(0.95).add_(p.grad); g_ = p.grad + 0.95 * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:                             # one batched Newton-Schulz per shape
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]; p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            for p, e, e2 in zip(rest, ex, ex2):
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-OTHER_LR * f)
    with torch.no_grad(): return m.loss(ev).item()


def H_(c):
    p = c[c > 0].double() / c.sum(); return -(p * p.log2()).sum().item()


def measures(tr):
    raw = tr.to(torch.uint8).numpy().tobytes(); cnt = torch.bincount(tr.flatten(), minlength=V)
    big = torch.bincount((tr[:, :-1] * V + tr[:, 1:]).flatten(), minlength=V * V)
    srt = cnt.sort(descending=True).values; srt = srt[srt > 0][:128].double(); r = torch.arange(1, len(srt) + 1).double().log()
    lc = srt.log(); slope = ((r - r.mean()) * (lc - lc.mean())).sum() / ((r - r.mean()) ** 2).sum()
    _, rc = torch.unique(tr, dim=0, return_counts=True)
    a, b = tr[:, :T // 2 + 1].flatten(), tr[:, T // 2:].flatten(); bs = b[torch.randperm(len(b), generator=torch.Generator().manual_seed(0))]
    mi = lambda u, w: H_(torch.bincount(u, minlength=V)) + H_(torch.bincount(w, minlength=V)) - H_(torch.bincount(u * V + w, minlength=V * V))
    torch.manual_seed(0); m = LM(); mats, _ = split(m); cs = []
    for j in range(4):
        gs = []
        for x in (tr[2 * j * B:(2 * j + 1) * B], tr[(2 * j + 1) * B:(2 * j + 2) * B]):
            m.zero_grad(); m.loss(x).backward(); gs.append(torch.cat([p.grad.flatten() for p in mats]))
        cs.append(F.cosine_similarity(gs[0], gs[1], 0).item())
    cos = sum(cs) / len(cs)
    return dict(gzip=len(gzip.compress(raw, 6)) / len(raw), h1=H_(cnt), h2=H_(big) - H_(cnt), zipf=-slope.item(),
                rare=cnt[cnt < 0.1 * cnt.sum() / V].sum().item() / cnt.sum().item(), dup=rc[rc > 1].sum().item() / len(tr),
                lrmi=mi(a, b) - mi(a, bs), gcos=cos, noise=B * (1 / max(cos, 1e-4) - 1))


_cache = {}
def job(a):
    name, lr, seed = a
    if name not in _cache: _cache[name] = torch.load(f"{HERE}/data{SUF}/{name}.pt")
    t0 = time.time(); l = train(_cache[name], lr, seed); return dict(ds=name, lr=lr, seed=seed, loss=l, sec=round(time.time() - t0, 2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        os.makedirs(f"{HERE}/data{SUF}", exist_ok=True); ms = {}
        for name, prm in datasets().items():
            if len(sys.argv) > 2 and name not in sys.argv[2].split(","): continue
            tr, ev = make(prm, NTR, 1), make(prm, NEV, 2); torch.save(dict(train=tr, eval=ev), f"{HERE}/data{SUF}/{name}.pt")
            ms[name] = dict(prm=prm, **measures(tr)); print(name, {k: round(v, 3) for k, v in ms[name].items() if k != "prm"}, flush=True)
        json.dump(ms, open(f"{HERE}/measures{SUF}.json", "w"), indent=1)
    elif cmd == "one":
        print(job((sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))))
    elif cmd == "sweep":
        import multiprocessing as mp
        names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(datasets())
        done = set()
        if os.path.exists(f"{HERE}/runs{SUF}.jsonl"):
            for l in open(f"{HERE}/runs{SUF}.jsonl"): r = json.loads(l); done.add((r["ds"], r["lr"], r["seed"]))
        jobs = [(n, lr, s) for n in names for lr in LRS for s in range(3) if (n, lr, s) not in done]
        t0 = time.time()
        with mp.get_context("spawn").Pool(int(sys.argv[2])) as pool, open(f"{HERE}/runs{SUF}.jsonl", "a") as f:
            for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
                f.write(json.dumps(r) + "\n"); f.flush()
        print(f"{len(jobs)} runs in {time.time() - t0:.0f}s")
    elif cmd == "fit":
        ms = json.load(open(f"{HERE}/measures{SUF}.json")); runs = {}
        for l in open(f"{HERE}/runs{SUF}.jsonl"): r = json.loads(l); runs.setdefault(r["ds"], {}).setdefault(r["lr"], []).append(r["loss"])
        best = {}
        for n, d in runs.items():
            lrs = sorted(d); L = [sum(d[x]) / len(d[x]) for x in lrs]; i = min(range(len(L)), key=L.__getitem__)
            if 0 < i < len(L) - 1:                                       # parabola through 3 points in log lr
                x0, x1, x2 = (math.log(lrs[j]) for j in (i - 1, i, i + 1)); y0, y1, y2 = L[i - 1], L[i], L[i + 1]
                den = (x0 - x1) * (x0 - x2) * (x1 - x2); A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
                Bq = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
                best[n] = dict(lr=math.exp(-Bq / (2 * A)) if A > 0 else lrs[i], loss=L[i], edge=False)
            else: best[n] = dict(lr=lrs[i], loss=L[i], edge=True)
        names = [n for n in best if n in ms]
        print(f"{'dataset':8} {'best lr':>8} {'loss':>6}  " + " ".join(f"{k:>6}" for k in ("gzip", "h2", "zipf", "rare", "dup", "lrmi", "noise")))
        for n in sorted(names, key=lambda n: best[n]["lr"]):
            print(f"{n:8} {best[n]['lr']:8.4f} {best[n]['loss']:6.3f}{'*' if best[n]['edge'] else ' '} " + " ".join(f"{ms[n][k]:6.3f}" if k != "noise" else f"{ms[n][k]:6.0f}" for k in ("gzip", "h2", "zipf", "rare", "dup", "lrmi", "noise")))
        ok = [n for n in names if not best[n]["edge"]]; y = torch.tensor([math.log(best[n]["lr"]) for n in ok], dtype=torch.double)
        rank = lambda v: v.argsort().argsort().double()
        print(f"\n{len(ok)} datasets with interior best lr; spread of best lr {math.exp(y.max() - y.min()):.1f}x")
        print("measure   spearman   LOO error of log-lr fit (x)   [baseline: predict the mean -> "
              f"{math.exp((y - y.mean()).abs().mean() * len(ok) / (len(ok) - 1)):.2f}x]")
        for k in ("gzip", "h1", "h2", "zipf", "rare", "dup", "lrmi", "gcos", "noise", "loss"):
            x = torch.tensor([best[n]["loss"] if k == "loss" else (math.log(ms[n][k]) if k == "noise" else ms[n][k]) for n in ok], dtype=torch.double)
            rho = torch.corrcoef(torch.stack([rank(x), rank(y)]))[0, 1].item(); err = []
            for j in range(len(ok)):
                msk = torch.arange(len(ok)) != j; X = torch.stack([x[msk], torch.ones(len(ok) - 1, dtype=torch.double)], 1)
                w = torch.linalg.lstsq(X, y[msk, None]).solution[:, 0]; err.append(abs(w[0] * x[j] + w[1] - y[j]).item())
            print(f"{k:8} {rho:8.2f}   {math.exp(sum(err) / len(err)):.2f}x")
