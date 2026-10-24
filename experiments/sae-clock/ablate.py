"""Remove one SAE latent from the model's residual stream (subtract activation x decoder vector, keep the SAE's error)
and see which answers break. usage: P=47 python3 ablate.py MODEL.pt SAE.pt"""
import sys, torch
import mod, sae
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); S = torch.load(sys.argv[2]); A, x, y = sae.acts(m); X = A["res"]
Z = sae.encode(X, S["We"], S["be"], S["bd"], S["k"]); g = torch.zeros(P, Z.shape[1]).index_add_(0, y, Z) / P
alive = (Z > 0).any(0); sup = (g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0); ans = g.argmax(0)
with torch.no_grad(): base = sae.head(m, X).argmax(-1) == y
rows = []
for j in alive.nonzero()[:, 0].tolist():
    with torch.no_grad(): pred = sae.head(m, X - Z[:, j:j + 1] * S["Wd"][:, j]).argmax(-1)
    ok = pred == y; tgt = g[:, j] > 0.3 * g[:, j].max(); on = tgt[y]           # pairs whose answer the latent codes
    rows.append((j, sup[j].item(), (base & on & ~ok).sum().item() / max(1, (base & on).sum().item()),
                 (base & ~on & ~ok).sum().item() / max(1, (base & ~on).sum().item()), pred[on & ~ok]))
one = [r for r in rows if r[1] == 1]; multi = [r for r in rows if r[1] > 1]
for name, rs in (("one-answer", one), ("multi-answer", multi)):
    if not rs: continue
    t = torch.tensor([r[2] for r in rs]); o = torch.tensor([r[3] for r in rs])
    print(f"{name:12} latents {len(rs)}: its answers break on {t.mean():.2f} of their pairs (median {t.median():.2f}); other answers break {o.mean():.4f}")
for j, sp, t, o, wrong in sorted(one, key=lambda r: -r[2])[:5]:
    vals, cnt = wrong.unique(return_counts=True); top = cnt.argsort(descending=True)[:3]
    print(f"  latent {j} (answer {ans[j].item()}): breaks {t:.2f}; wrong answers instead {vals[top].tolist()} x {cnt[top].tolist()}")
