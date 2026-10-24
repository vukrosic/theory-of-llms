"""Is there one vector that carries the task, and can it be patched into another prompt?
Trains icl.py's model (env as icl.py), then per layer l (-1 = embedding output, L-1 = last block):
  v_t = mean hidden state at the last x position over few-shot prompts of task t, minus the mean over all tasks
  zero  = accuracy of f_t(x) when v_t is added to the hidden state of a lone x (no examples at all)
  swap  = a few-shot prompt of task u gets v_t - v_u added at the last x: accuracy of f_t(x) (the patched task wins)
base: zero-shot without patch (no way to know the task) and the full few-shot accuracy at the last pair
usage: python3 tv.py SEED STEPS LR -> json row in tv_runs.jsonl"""
import os, sys, json, time, torch
import icl
seed, steps, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
t0 = time.time(); m = icl.train(seed, steps, lr); m.eval(); T, K, P = icl.T, icl.K, icl.P
g = torch.Generator().manual_seed(77); N = 64                     # prompts per task for the mean
H = {}
def grab(l, h): H[l] = h[:, -1].clone(); return h
with torch.no_grad():
    vec = []                                                       # (T, L+1, D)
    for t in range(T):
        s, _ = icl.gen(N, g, task=torch.full((N,), t)); s = s[:, :-1]     # ends on the last x
        m.hook = grab; m(s); vec.append(torch.stack([H[l].mean(0) for l in range(-1, icl.L)]))
    vec = torch.stack(vec); vec = vec - vec.mean(0, keepdim=True)
    x = torch.arange(P).repeat(T); tt = torch.arange(T).repeat_interleave(P)   # every (task, x) pair once
    want = icl.POOL[tt, x] + P
    m.hook = None; base_zero = (m(x[:, None])[:, -1].argmax(-1) == want).float().mean().item()
    s, _ = icl.gen(len(x), g, task=tt); s = torch.cat([s[:, :-2], x[:, None]], 1)          # K-1 pairs of task tt, then x
    few = (m(s)[:, -1].argmax(-1) == want).float().mean().item()
    u = (tt + 1 + torch.randint(0, T - 1, tt.shape, generator=g)) % T                    # a different task for the swap prompt
    su, _ = icl.gen(len(x), g, task=u); su = torch.cat([su[:, :-2], x[:, None]], 1)
    zero, swap, swap_u = [], [], []
    A = float(os.environ.get("ALPHA", 1))   # patch strength
    for j, l in enumerate(range(-1, icl.L)):
        def add0(li, h, j=j, l=l):
            if li == l: h = h.clone(); h[:, -1] += A * vec[tt, j]
            return h
        m.hook = add0; zero.append(round((m(x[:, None])[:, -1].argmax(-1) == want).float().mean().item(), 3))
        def addsw(li, h, j=j, l=l):
            if li == l: h = h.clone(); h[:, -1] += A * (vec[tt, j] - vec[u, j])
            return h
        m.hook = addsw; o = m(su)[:, -1].argmax(-1)
        swap.append(round((o == want).float().mean().item(), 3)); swap_u.append(round((o == icl.POOL[u, x] + P).float().mean().item(), 3))
    m.hook = None
    # how many directions do the task vectors use: share of their variance in the top 1, 2, 4 directions, per layer
    sv = [torch.linalg.svdvals(vec[:, j]) ** 2 for j in range(icl.L + 1)]
    top = [[round((s_[:k].sum() / s_.sum()).item(), 2) for k in (1, 2, 4)] for s_ in sv]
row = dict(ALPHA=A, T=T, BURST=icl.BURST, D=icl.D, L=icl.L, seed=seed, steps=steps, lr=lr, few=round(few, 3), base_zero=round(base_zero, 3),
           zero=zero, swap=swap, swap_keeps_old=swap_u, top124=top, sec=round(time.time() - t0, 1))
print(json.dumps(row)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv_runs.jsonl"), "a").write(json.dumps(row) + "\n")
