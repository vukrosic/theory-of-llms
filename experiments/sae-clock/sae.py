"""Train a sparse autoencoder on the clock adder's activations at the '=' position and label every latent.
usage: P=47 python3 sae.py MODEL.pt SITE N K SEED [STEPS]
  SITE res = residual stream after the block (64 dims, what the head reads), mlp = MLP neurons (128)
  N latents, K = active latents per input (TopK SAE, Gao et al. 2024); K=0 -> ReLU + L1 (L1 coef from env L1, default 3e-3)
Labels: each latent's 47 x 47 map over (a, b) is checked for being a function of a alone, b alone or (a+b) mod P,
and that function's main frequency (its clock speed) and phase (where on the clock it fires).
Saves sae_runs/<model>_<site>_<N>_<K>_<seed>.pt with weights + labels, appends a line to runs.jsonl.
"""
import json, math, os, sys, time
import torch, torch.nn.functional as F
import mod
P = mod.P; torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def acts(m):
    x, y, _, _ = mod.data(0); B = m.blk
    h = m.emb(x); r1 = h + B.attn(B.n1(h)); z = B.n2(r1); hid = F.silu(B.gate(z)) * B.up(z); r2 = r1 + B.down(hid)
    return dict(res=r2[:, -1], mlp=hid[:, -1], r1=r1[:, -1], z=z[:, -1], out=B.down(hid)[:, -1]), x, y


def head(m, r):                                            # answer scores from a residual vector at '='
    return m.head(m.norm(r))


def train(X, n, k, seed, steps, l1=3e-3, Y=None, skip=False):   # Y given: a transcoder (read X, write Y), else an SAE (Y = X); skip: + linear X -> Y path (skip transcoder)
    torch.manual_seed(seed); d = X.shape[1]; tc = Y is not None; Y = X if Y is None else Y
    Wd = F.normalize(torch.randn(Y.shape[1], n), dim=0); We = (Wd.T.clone() if not tc else F.normalize(torch.randn(n, d), dim=1)).requires_grad_()
    Wd = Wd.requires_grad_(); be = torch.zeros(n, requires_grad=True); bd = Y.mean(0).clone().requires_grad_()
    bi = X.mean(0) if tc else None; Ws = torch.zeros(Y.shape[1], d, requires_grad=True)
    opt = torch.optim.Adam([We, Wd, be, bd] + ([Ws] if skip else []), lr=float(os.environ.get("SAE_LR", 3e-3)))
    enc = lambda X_: encode(X_, We, be, bi if tc else bd, k)
    for i in range(steps):
        Z = enc(X); R = Z @ Wd.T + bd + ((X - bi) @ Ws.T if skip else 0); loss = (R - Y).pow(2).sum(1).mean()
        if k == 0: loss = loss + l1 * Z.sum(1).mean() * X.pow(2).sum(1).mean().sqrt()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Wd.data = F.normalize(Wd.data, dim=0)
    if tc and skip: return We.detach(), be.detach(), Wd.detach(), bd.detach(), bi, Ws.detach()
    if tc: return We.detach(), be.detach(), Wd.detach(), bd.detach(), bi
    return We.detach(), be.detach(), Wd.detach(), bd.detach()


def encode(X, We, be, bd, k):
    pre = F.relu((X - bd) @ We.T + be)
    if k == 0: return pre
    v, i = pre.topk(k, -1); return torch.zeros_like(pre).scatter(-1, i, v)


def label(Z):                                              # Z: P*P x n latent activations
    M = Z.view(P, P, -1); M = M - M.mean((0, 1)); tot = M.pow(2).sum((0, 1)).clamp_min(1e-12)
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij"); s = (a + b) % P
    fa = M.mean(1); fb = M.mean(0)
    gs = torch.zeros(P, M.shape[-1]).index_add_(0, s.flatten(), M.reshape(P * P, -1)) / P
    shares = torch.stack([fa.pow(2).sum(0) * P, fb.pow(2).sum(0) * P, gs.pow(2).sum(0) * P]) / tot   # a alone, b alone, a+b
    typ = shares.argmax(0); prof = torch.stack([fa, fb, gs])[typ, :, torch.arange(M.shape[-1])]          # n x P
    Fq = torch.fft.rfft(prof, dim=1); pw = Fq.abs().pow(2)[:, 1:]; kk = pw.argmax(1) + 1
    purity = pw.max(1).values / pw.sum(1).clamp_min(1e-12)
    phase = torch.atan2(Fq[torch.arange(len(kk)), kk].imag, Fq[torch.arange(len(kk)), kk].real)
    H2 = torch.fft.fft2(M, dim=(0, 1)).abs() ** 2; best2, k2 = torch.zeros(M.shape[-1]), torch.zeros(M.shape[-1], dtype=torch.long)
    for f in range(1, P // 2 + 1):                         # 2D: power on a, b, a+b, a-b at clock f (both signs), any mix of a and b
        idx = [(f, 0), (P - f, 0), (0, f), (0, P - f), (f, f), (P - f, P - f), (f, P - f), (P - f, f)]
        sh = sum(H2[i, j] for i, j in idx) / (tot * P * P); k2 = torch.where(sh > best2, f, k2); best2 = torch.maximum(best2, sh)
    return dict(clock2d=k2, share2d=best2, type=typ, share=shares.max(0).values, k=kk, purity=purity, phase=phase, alive=(Z > 0).float().mean(0))


def split(V):                                              # V = f(a) + g(b)  +  q((a+b) mod P)  +  rest
    V = V.view(P, P, -1); V = V - V.mean((0, 1)); fa = V.mean(1, keepdim=True); gb = V.mean(0, keepdim=True)
    W = V - fa - gb; s = (torch.arange(P)[:, None] + torch.arange(P)[None]) % P
    Q = (torch.zeros(P, V.shape[-1]).index_add_(0, s.flatten(), W.reshape(P * P, -1)) / P)[s]
    return (fa + gb).expand_as(V), Q, W - Q


def kept(X, R):                                            # share of each part of X that R rebuilds
    return [round((1 - (a - b).pow(2).sum() / a.pow(2).sum()).item(), 3) for a, b in zip(split(X), split(R))]


if __name__ == "__main__":
    path, site, n, k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
    m = mod.Model(); m.load_state_dict(torch.load(path)); A, x, y = acts(m); X = A[site]
    t0 = time.time(); We, be, Wd, bd = train(X, n, k, seed, steps, float(os.environ.get("L1", 3e-3)))
    Z = encode(X, We, be, bd, k); R = Z @ Wd.T + bd
    fvu = ((R - X).pow(2).sum() / (X - X.mean(0)).pow(2).sum()).item()
    acc = lambda r: (head(m, r).argmax(-1) == y).float().mean().item()
    sp = (acc(R), acc(X)) if site == "res" else (acc(A["r1"] + m.blk.down(R)), acc(A["r1"] + m.blk.down(X)))
    kp = kept(X, R); print(f"  rebuilt: a alone + b alone {kp[0]:.2f}, the answer (a+b) {kp[1]:.2f}, the rest {kp[2]:.2f}")
    L = label(Z); alive = L["alive"] > 0
    T = ["a", "b", "a+b"]
    print(f"{os.path.basename(path)} {site} n={n} k={k} seed={seed} {time.time() - t0:.1f}s  unexplained var {fvu:.3f}  "
          f"acc with SAE output {sp[0]} (model {sp[1]})  alive {alive.sum().item()}/{n}  L0 {(Z > 0).float().sum(1).mean():.1f}")
    o = L["share"].argsort(descending=True)
    groups = {}
    for j in o[alive[o]].tolist():
        groups.setdefault((T[L["type"][j]], L["k"][j].item()), []).append(j)
    sm = (x[:, 0] + x[:, 1]) % P; g = torch.zeros(P, n).index_add_(0, sm, Z) / P
    sup = (g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) * alive
    one = (sup == 1); covered = len(set(g[:, one].argmax(0).tolist()))
    print(f"  latents firing on one answer only: {one.sum().item()} of {alive.sum().item()} alive, covering {covered} of {P} answers;"
          f" answers per latent {torch.bincount(sup[alive], minlength=6)[1:6].tolist()} (1..5)")
    clean = (L["share"] > 0.8) & (L["purity"] > 0.5) & alive
    c2 = (L["share2d"] > 0.5) & alive; bc = torch.bincount(L["clock2d"][c2], minlength=P // 2 + 1)
    print(f"  latents with >50% of their (a, b) map on one clock: {c2.sum().item()} of {alive.sum().item()}; clocks "
          + " ".join(f"{f}:{bc[f].item()}" for f in bc.argsort(descending=True)[:8].tolist() if bc[f] > 0) + f"; median share {L['share2d'][alive].median():.2f}")
    print(f"  latents that are one function of a, b or a+b (>80%) with one main clock (>50%): {clean.sum().item()} of {alive.sum().item()} alive")
    for (t, kk), js in sorted(groups.items(), key=lambda g: -len(g[1]))[:int(os.environ.get("SHOW", 0))]:
        print(f"  {t:3} clock {kk:2}: {len(js):2} latents  share {L['share'][js].mean():.2f} purity {L['purity'][js].mean():.2f}  "
              f"phases {' '.join(f'{L['phase'][j]:+.1f}' for j in js[:8])}")
    os.makedirs(os.path.join(HERE, "sae_runs"), exist_ok=True)
    tag = f"{os.path.basename(path)[:-3]}_{site}_{n}_{k}_{seed}"
    torch.save(dict(We=We, be=be, Wd=Wd, bd=bd, k=k, labels=L), os.path.join(HERE, "sae_runs", tag + ".pt"))
    with open(os.path.join(HERE, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(tag=tag, fvu=round(fvu, 4), acc=sp[0], kept=kp, alive=alive.sum().item(), clean=clean.sum().item(), one=one.sum().item(), covered=covered,
                                 groups={f"{t} {kk}": len(js) for (t, kk), js in groups.items()})) + "\n")
