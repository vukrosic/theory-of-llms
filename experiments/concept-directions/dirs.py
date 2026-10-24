"""Concept directions: which concepts of the 8-skill toy are straight lines inside the model, and does adding a direction steer the output.
usage: MOE=8 MSIGN=ns EMB_MULT=8 [WIDTH=64] python3 dirs.py SEED [STEPS] [LR]      appends one row to runs.jsonl

concepts (each read at the position that predicts the answer)
  count  how many 0s (0-CMAX, default 4), balanced inputs                 ordered, 7 values
  slot   which sorted output is next (0-5) in the sort task   ordered, 6 values
  rel    fact relation: common-answer facts (factc) vs rare-answer facts (factr), same 64 subjects
sites: 0 = embedding output, 1 = after layer 0, 2 = after layer 1 (into the final norm + head)
per concept and site
  lin / mlp   held-out accuracy of a linear and a 1-hidden-layer probe (the gap = how much is not linear)
  pc1         share of the class means' spread on their top direction (1 = all means on one straight line)
  turn        mean cosine between consecutive steps mean(c+1)-mean(c) (1 = straight, <1 = bent)
steering (only inputs the model gets right, target answer different from the source answer), success = argmax is the target answer
  chord   add mean(target) - mean(source)                   (needs examples of the exact target)
  line    add delta x the best-line step vector              (one direction for the whole concept)
  first   add delta x (mean(1) - mean(0))                    (one measured step, reused)
  rand    random vector with the chord's size                (control)
  each at strength 1 and the best of 0.5, 1, 2, 4; rel: chord built from 32 subjects, tested on the other 32
"""
import sys, os, json, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../muon/toy-suite"))
import torch, torch.nn.functional as F
import suite, mix, sampled
from sampled import SKILLS, NV, SEP, NC, SUBJ, FACTC, FACTR, L

seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 260; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.014
DRIFT = os.environ.get("DRIFT", "0") == "1"           # also keep checkpoints at 1/4, 1/2, 3/4 of training and test their directions + probes on the final model
SNAP, snaps = ([int(f * steps) for f in (0.25, 0.5, 0.75)] if DRIFT else []), {}
torch.manual_seed(seed); model = suite.Model(); _lr_at = suite.lr_at
def lr_at(i, n):
    if i in SNAP and i not in snaps: snaps[i] = copy.deepcopy(model.state_dict())
    return _lr_at(i, n)
suite.lr_at = lr_at
t0 = time.time(); rows = mix.run("whole", seed, steps, lr, every=steps, bs=32, init=model); m = mix.last_model.eval(); t_train = time.time() - t0
acc = {sk: round(rows[-1][1][sk][1], 3) for sk in SKILLS}
g = torch.Generator().manual_seed(1000 + seed); PAD = sampled.PAD
tok = lambda sk: NV + 2 + SKILLS.index(sk)


CMAX = int(os.environ.get("CMAX", 4))                 # counts 5-6 are almost never seen in training (0.4%)
def count_data(n):                                   # n inputs per count 0..CMAX
    xs, ys = [], []
    for c in range(CMAX + 1):
        a = torch.randint(1, 4, (n, NC), generator=g); idx = torch.rand(n, NC, generator=g).argsort(1)[:, :c]; a.scatter_(1, idx, 0)
        s = torch.full((n, L + 1), PAD); s[:, 0], s[:, 1:NC + 1], s[:, NC + 1], s[:, NC + 2] = tok("count"), a, SEP, c
        xs.append(s); ys.append(torch.full((n,), c))
    s = torch.cat(xs); return s[:, :-1], torch.full((len(s),), NC + 1), torch.cat(ys), s[:, NC + 2]


def slot_data(n):                                    # every sort input gives 6 points, one per output slot
    s, _ = sampled.gen("sort", n, g); x = s[:, :-1].repeat_interleave(6, 0); j = torch.arange(6).repeat(n)
    return x, 7 + j, j, s[:, 8:14].repeat_interleave(6, 0)[torch.arange(6 * n), j], s[:, 8:14].repeat_interleave(6, 0)


def rel_data(n):
    xs, ys, ans, subj = [], [], [], []
    for r, sk in enumerate(("factc", "factr")):
        j = torch.randint(64, (n,), generator=g); s = torch.full((n, L + 1), PAD); s[:, 0], s[:, 1], s[:, 2] = tok(sk), SUBJ[j], SEP
        xs.append(s); ys.append(torch.full((n,), r)); subj.append(j)
    s = torch.cat(xs); j = torch.cat(subj); return s[:, :-1], torch.full((len(s),), 2), torch.cat(ys), j


ADD = {}                                             # site -> (positions, vectors) added in the forward pass
def hook(site):
    def f(mod, i, o):
        HID[site] = o
        if site in ADD:
            p, v = ADD[site]; o = o.clone(); o[torch.arange(len(o)), p] += v
        return o
    return f
HID = {}
hs = [m.emb.register_forward_hook(hook(0))] + [l.register_forward_hook(hook(i + 1)) for i, l in enumerate(m.layers)]


@torch.no_grad()
def run(x, p, site=None, v=None):
    ADD.clear()
    if site is not None: ADD[site] = (p, v)
    out = m(x)[torch.arange(len(x)), p]
    return out.argmax(-1), {k: h[torch.arange(len(x)), p].clone() for k, h in HID.items()}


def probe(X, y, hidden, it=200, Xt=None):
    """held-out accuracy (70/30 split); with Xt: train on all of X, test on Xt (same rows, another checkpoint)"""
    n = len(y); perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); a, b = perm[: n * 7 // 10], perm[n * 7 // 10:]
    if Xt is not None: a = b = torch.arange(n)
    mu, sd = X[a].mean(0), X[a].std(0) + 1e-6; Xn = (X - mu) / sd; k = int(y.max()) + 1
    net = torch.nn.Linear(X.shape[1], k) if not hidden else torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, k))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(it):
        opt.zero_grad(); F.cross_entropy(net(Xn[a]), y[a]).backward(); opt.step()
    with torch.no_grad(): return (net(((Xt - mu) / sd if Xt is not None else Xn)[b]).argmax(-1) == y[b]).float().mean().item()


def geometry(X, y):
    mus = torch.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])
    S = torch.linalg.svdvals(mus - mus.mean(0)) ** 2; st = mus[1:] - mus[:-1]
    turn = F.cosine_similarity(st[1:], st[:-1], dim=1).mean().item() if len(st) > 1 else 1.0
    return mus, (S[0] / S.sum()).item(), turn


STR = [0.5, 1, 2, 4]
def steer_ordinal(x, p, y, ans_of, site, mus, ok, meths=("chord", "line", "first", "rand")):
    """ans_of(rows, target value) -> correct answer token for those rows at the target value"""
    c = torch.arange(len(mus)).float(); line = ((c - c.mean())[:, None] * (mus - mus.mean(0))).sum(0) / (c - c.mean()).pow(2).sum()
    first = mus[1] - mus[0]; res = {}
    for meth in meths:
        for a_ in STR:
            hit, tot = {}, {}
            for d in (-2, -1, 1, 2):
                src = ok & (y + d >= 0) & (y + d < len(mus)); r = src.nonzero()[:, 0]
                if len(r) == 0: continue
                tgt = y[r] + d; want = ans_of(r, tgt); keep = want != ans_of(r, y[r]); r, tgt, want = r[keep], tgt[keep], want[keep]
                if len(r) == 0: continue
                v = mus[tgt] - mus[y[r]] if meth == "chord" else d * line.expand(len(r), -1) if meth == "line" else d * first.expand(len(r), -1)
                if meth == "rand":
                    rv = torch.randn(v.shape, generator=torch.Generator().manual_seed(7)); v = rv / rv.norm(dim=1, keepdim=True) * (mus[tgt] - mus[y[r]]).norm(dim=1, keepdim=True)
                pred, _ = run(x[r], p[r], site, a_ * v); hit[d] = (pred == want).float().mean().item(); tot[d] = len(r)
            res[(meth, a_)] = hit
    out = {}
    for meth in meths:
        mean = lambda a_: sum(res[(meth, a_)].values()) / max(1, len(res[(meth, a_)]))
        best = max(STR, key=mean)
        out[meth] = dict(s1={d: round(v, 3) for d, v in res[(meth, 1)].items()}, best_a=best, best={d: round(v, 3) for d, v in res[(meth, best)].items()})
    return out


row = dict(seed=seed, steps=steps, lr=lr, width=suite.D, moe=suite.MOE, mix=sampled.MIX, acc=acc, t_train=round(t_train, 1), concepts={})
CONCEPTS = os.environ.get("CONCEPTS", "count,rel").split(","); KEEP = {}   # slot is off by default: the toy learns sort only to ~50%
DATA = dict(count=lambda: (lambda d: (d[0], d[1], d[2], d[3]))(count_data(160)), slot=lambda: (lambda d: (d[0], d[1], d[2], d[4]))(slot_data(160)), rel=lambda: rel_data(256))
for name in CONCEPTS:
    x, p, y, extra = DATA[name]()
    pred, H = run(x, p)
    if name == "count": right = extra; ans_of = lambda r, t: t
    elif name == "slot": right = extra[torch.arange(len(x)), y]; ans_of = lambda r, t, E=extra: E[r, t]
    else:
        right = torch.where(y == 1, FACTR[extra], FACTC[extra]); ans_of = None
    ok = pred == right; c = dict(base_acc=round(ok.float().mean().item(), 3), sites={})
    for site in (1, 2):                              # site 0 (embedding of the separator) is the same for every input
        X = H[site]; mus, pc1, turn = geometry(X, y)
        sd = dict(lin=round(probe(X, y, False), 3), mlp=round(probe(X, y, True), 3), pc1=round(pc1, 3), turn=round(turn, 3))
        if name == "count" and site == 2:            # cause check: do the last-layer count means point at their own output rows?
            gm = (mus - mus.mean(0)) * m.norm.weight.detach(); Wc = m.head.weight.detach()[:len(mus)]; Wc = Wc - Wc.mean(0)
            sd["head_align"] = round(F.cosine_similarity(gm, Wc, dim=1).mean().item(), 3)
            ws = Wc[1:] - Wc[:-1]; sd["head_turn"] = round(F.cosine_similarity(ws[1:], ws[:-1], dim=1).mean().item(), 3)
        if name in ("count", "slot"): sd["steer"] = steer_ordinal(x, p, y, ans_of, site, mus, ok)
        else:                                        # relation: factc -> factr and back, chord from subjects 0-31, tested on 32-63
            st = {}
            for src, tg in ((0, 1), (1, 0)):
                fit = extra < 32; vec = X[(y == tg) & fit].mean(0) - X[(y == src) & fit].mean(0)
                r = ((y == src) & ~fit & ok).nonzero()[:, 0]; want = (FACTR if tg == 1 else FACTC)[extra[r]]
                rv = torch.randn(vec.shape, generator=torch.Generator().manual_seed(7)); rv = rv / rv.norm() * vec.norm()
                hits = {a_: (run(x[r], p[r], site, a_ * vec.expand(len(r), -1))[0] == want).float().mean().item() for a_ in STR}
                rh = {a_: (run(x[r], p[r], site, a_ * rv.expand(len(r), -1))[0] == want).float().mean().item() for a_ in STR}
                bst = max(STR, key=hits.get)
                st[f"{src}to{tg}"] = dict(s1=round(hits[1], 3), best_a=bst, best=round(hits[bst], 3), rand_best=round(max(rh.values()), 3), n=len(r))
            sd["steer"] = st
        c["sites"][site] = sd
    row["concepts"][name] = c; KEEP[name] = (x, p, y, extra, ok, H)


def rel_vec(X, y, extra, src, tg): fit = extra < 32; return X[(y == tg) & fit].mean(0) - X[(y == src) & fit].mean(0)
def rel_hit(x, p, y, extra, ok, site, vec, src, tg):
    r = ((y == src) & (extra >= 32) & ok).nonzero()[:, 0]; want = (FACTR if tg == 1 else FACTC)[extra[r]]
    return {a_: (run(x[r], p[r], site, a_ * vec.expand(len(r), -1))[0] == want).float().mean().item() for a_ in STR}


if DRIFT:                                            # directions and probes measured at an earlier checkpoint, used on the final model
    final = copy.deepcopy(m.state_dict()); row["drift"] = {}
    for i, sd_ in snaps.items():
        m.load_state_dict(sd_); old = {nm: run(KEEP[nm][0], KEEP[nm][1])[1] for nm in KEEP}; m.load_state_dict(final); dr = {}
        for site in (1, 2):
            e = {}
            if "count" in KEEP:
                x, p, y, extra, ok, H = KEEP["count"]; mo, mf = geometry(old["count"][site], y)[0], geometry(H[site], y)[0]
                st_ = steer_ordinal(x, p, y, lambda r, t: t, site, mo, ok, meths=("chord",))["chord"]
                cs = [F.cosine_similarity(mo[a] - mo[b], mf[a] - mf[b], dim=0).item() for a in range(len(mo)) for b in range(a)]
                e["count"] = dict(chord_s1=round(sum(st_["s1"].values()) / len(st_["s1"]), 3), chord_best=round(sum(st_["best"].values()) / len(st_["best"]), 3),
                                  cos=round(sum(cs) / len(cs), 3), probe=round(probe(old["count"][site], y, False, Xt=H[site]), 3))
            if "rel" in KEEP:
                x, p, y, extra, ok, H = KEEP["rel"]; hh = []
                for src, tg in ((0, 1), (1, 0)): hh.append(max(rel_hit(x, p, y, extra, ok, site, rel_vec(old["rel"][site], y, extra, src, tg), src, tg).values()))
                cs = F.cosine_similarity(rel_vec(old["rel"][site], y, extra, 0, 1), rel_vec(H[site], y, extra, 0, 1), dim=0).item()
                e["rel"] = dict(best=round(sum(hh) / 2, 3), cos=round(cs, 3), probe=round(probe(old["rel"][site], y, False, Xt=H[site]), 3))
            dr[site] = e
        row["drift"][i] = dr
row["t_total"] = round(time.time() - t0, 1)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
print(f"width {suite.D} steps {steps} seed {seed}  train {t_train:.1f}s total {row['t_total']}s  acc " + " ".join(f"{k}:{v:.2f}" for k, v in acc.items()))
for name, c in row["concepts"].items():
    print(f" {name} base {c['base_acc']}")
    for site, sd in c["sites"].items():
        s = sd["steer"]
        if name == "rel": ss = " ".join(f"{k}: s1 {v['s1']} best {v['best']}@{v['best_a']} rand {v['rand_best']}" for k, v in s.items())
        else: ss = " | ".join(f"{k} s1 {sum(v['s1'].values()) / max(1, len(v['s1'])):.2f} best {sum(v['best'].values()) / max(1, len(v['best'])):.2f}@{v['best_a']}" for k, v in s.items())
        print(f"   site {site} lin {sd['lin']} mlp {sd['mlp']} pc1 {sd['pc1']} turn {sd['turn']}" + (f" head_align {sd['head_align']} head_turn {sd['head_turn']}" if "head_align" in sd else "") + f"  {ss}")
for i, dr in row.get("drift", {}).items():
    print(f" drift: directions from step {i} on the final model: " + " | ".join(f"site {site} " + " ".join(f"{nm} {v}" for nm, v in e.items()) for site, e in dr.items()))
