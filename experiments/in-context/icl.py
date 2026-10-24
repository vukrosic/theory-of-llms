"""In-context task toy. Each sequence shows K pairs x1 y1 ... xK yK of one task: a fixed random table f_t(x) from a pool of T tasks.
The model predicts each y_i at the x_i position. Two ways to be right:
  recog = the task was seen in training and x_i is NEW in this prompt: the model must tell which task it is from the earlier pairs
          and apply the table it stored in its weights (task recognition, what function vectors carry in LLMs)
  copy  = a NEVER-seen task and x_i appeared earlier in the prompt: the model must read the answer from the prompt (task learning)
env: T (task pool, 0 = a fresh random task every sequence), P inputs, Q outputs, K pairs, BURST (distinct x per sequence, 0 = all P),
     D width, L layers, ZIPF (task frequency exponent, 0 = uniform), WD (Muon weight decay)
usage: python3 icl.py SEED STEPS LR  -> prints json row
"""
import os, sys, json, time, math, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
E = os.environ.get
T, P, Q, K = int(E("T", 16)), int(E("P", 32)), int(E("Q", 32)), int(E("K", 12))
BURST, D, L = int(E("BURST", 0)) or P, int(E("D", 64)), int(E("L", 2))
H = max(1, D // 16); ZIPF, WD, BS = float(E("ZIPF", 0)), float(E("WD", 0)), int(E("BS", 32))
MIXB = int(E("MIXB", 0)) or BURST   # distinct x in each fresh-rule sequence (P = almost no repeats: nothing to read)
MIXF = float(E("MIXF", 0))   # share of training sequences that use a fresh random task (never repeats) instead of the pool
V = P + Q

def rope(x, base=10000):
    s, d = x.shape[-2], x.shape[-1]
    th = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(s).float(), th); c, sn = ang.cos(), ang.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * c - b * sn, b * c + a * sn], -1).flatten(-2)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        F_ = 176 * D // 64
        s.gate, s.up, s.down = nn.Linear(D, F_, bias=False), nn.Linear(D, F_, bias=False), nn.Linear(F_, D, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(D), nn.RMSNorm(D), nn.RMSNorm(D // H), nn.RMSNorm(D // H)
    def forward(s, h):
        b, t, _ = h.shape; x = s.n1(h)
        q, k, v = (f(x).view(b, t, H, D // H).transpose(1, 2) for f in (s.q, s.k, s.v))
        q, k = rope(s.qn(q)), rope(s.kn(k))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + s.o(a.transpose(1, 2).reshape(b, t, D))
        x = s.n2(h)
        return h + s.down(F.silu(s.gate(x)) * s.up(x))

class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D); s.blocks = nn.ModuleList([Block() for _ in range(L)])
        s.norm, s.head = nn.RMSNorm(D), nn.Linear(D, V, bias=False)
        s.hook = None   # hook(layer_index, h) -> h, called after each block (and with -1 after the embedding)
    def forward(s, x):
        h = s.emb(x)
        if s.hook: h = s.hook(-1, h)
        for i, bl in enumerate(s.blocks):
            h = bl(h)
            if s.hook: h = s.hook(i, h)
        return s.head(s.norm(h))

def tables(n, g): return torch.randint(0, Q, (n, P), generator=g)
POOL = tables(max(T, 1), torch.Generator().manual_seed(1234))   # the training tasks (same for every seed)
NOVEL = tables(4096, torch.Generator().manual_seed(4321))        # never-seen tasks for the copy test
W = (torch.arange(1, max(T, 1) + 1).float() ** -ZIPF); W /= W.sum()

def gen(n, g, tab=None, task=None, burst=BURST):
    """n sequences: returns tokens (n, 2K) and the task id per sequence. tab = tables to use (default the pool)."""
    if tab is None:
        if T == 0: tab = tables(n, g); task = torch.arange(n)
        else: tab = POOL; task = torch.multinomial(W, n, replacement=True, generator=g) if task is None else task
    elif task is None: task = torch.randint(0, len(tab), (n,), generator=g)
    sub = torch.rand(n, P, generator=g).argsort(1)[:, :burst]                      # the distinct x this sequence uses
    xs = sub.gather(1, torch.randint(0, burst, (n, K), generator=g))
    ys = tab[task][torch.arange(n)[:, None], xs] + P
    return torch.stack([xs, ys], 2).view(n, 2 * K), task

def seen_before(xs):   # (n, K) bool: x_i appeared at an earlier pair
    return (xs[:, :, None] == xs[:, None, :]).tril(-1).any(2)

def msign(g):
    a, b, c = 3.4445, -4.7750, 2.0315; tr = g.shape[-2] > g.shape[-1]
    X = g.mT if tr else g; X = X / (X.norm() + 1e-7)
    for _ in range(5): A = X @ X.mT; X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tr else X

def lr_at(i, steps): return min(1.0, (i + 1) / 30) * min(1.0, 2 * (1 - i / steps))

def accs(m, n=512):
    """recog: pool tasks, new x, pair index >= 4.  copy: novel tasks, repeated x.  rep: pool tasks, repeated x."""
    g = torch.Generator().manual_seed(999); out = {}
    with torch.no_grad():
        for name, tab in (("pool", None), ("novel", NOVEL)):
            s, _ = gen(n, g, tab) if T or tab is not None else gen(n, g)
            pred = m(s)[:, 0::2].argmax(-1); ok = pred == s[:, 1::2]; rep = seen_before(s[:, 0::2])
            late = torch.arange(K)[None] >= 4
            if name == "pool": out["recog"] = ok[~rep & late].float().mean().item(); out["rep"] = ok[rep].float().mean().item()
            else: out["copy"] = ok[rep].float().mean().item(); out["novel_new"] = ok[~rep & late].float().mean().item()
        if T:   # conflict: a pool task whose prompt shows NEW labels for 3 of its inputs (same new label at every repeat)
            n2 = n; task = torch.randint(0, T, (n2,), generator=g); tab = POOL[task].clone()
            ch = torch.rand(n2, P, generator=g).argsort(1)[:, :3]; new = (tab.gather(1, ch) + 1 + torch.randint(0, Q - 1, (n2, 3), generator=g)) % Q
            tab.scatter_(1, ch, new)
            s, _ = gen(n2, g, tab, torch.arange(n2)); xs = s[:, 0::2]
            pred = m(s)[:, 0::2].argmax(-1) - P; rep = seen_before(xs); changed = (xs[:, :, None] == ch[:, None, :]).any(2)
            sel = rep & changed; memo = POOL[task][torch.arange(n2)[:, None], xs]
            out["conf_ctx"] = (pred == s[:, 1::2] - P)[sel].float().mean().item(); out["conf_mem"] = (pred == memo)[sel].float().mean().item()
    return out

PRE = int(E("PRE", 0))   # >0: first train PRE steps on fresh rules only (builds prompt reading), then the given steps on the pool (fine-tuning)

def train(seed, steps, lr, every=0):
    global T
    torch.manual_seed(seed); m = Model()
    if PRE:
        t_, T = T, 0; m = train_on(m, seed + 100, PRE, lr, 0); T = t_
    return train_on(m, seed, steps, lr, every)

def train_on(m, seed, steps, lr, every=0):
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and "blocks" in n]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; g = torch.Generator().manual_seed(seed); curve = []
    for i in range(steps):
        if every and i % every == 0: curve.append((i, accs(m, 256)))
        s, _ = gen(BS, g)
        if MIXF:
            nf = int((torch.rand(BS, generator=g) < MIXF).sum())
            if nf: s[:nf] = gen(nf, g, tables(nf, g), burst=MIXB)[0]
        loss = F.cross_entropy(m(s)[:, 0::2].reshape(-1, V), s[:, 1::2].reshape(-1))
        f = lr_at(i, steps)
        for gr in adam.param_groups: gr["lr"] = 0.01 * f
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b in zip(mats, buf):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                if WD: p.mul_(1 - lr * f * WD)
                p.add_(msign(u), alpha=-lr * f * max(1, p.shape[-2] / p.shape[-1]) ** 0.5)
    m.curve = curve; m.loss = loss.item()
    return m

if __name__ == "__main__":
    seed, steps, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    t0 = time.time(); m = train(seed, steps, lr, int(E("EVERY", 0))); a = accs(m)
    row = dict(MIXB=MIXB, PRE=PRE, MIXF=MIXF, T=T, P=P, Q=Q, K=K, BURST=BURST, D=D, L=L, ZIPF=ZIPF, WD=WD, seed=seed, steps=steps, lr=lr, loss=round(m.loss, 3),
               **{k: round(v, 3) for k, v in a.items()}, curve=[(i, round(c['recog'], 2), round(c['copy'], 2)) for i, c in m.curve], sec=round(time.time() - t0, 1))
    print(json.dumps(row))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
