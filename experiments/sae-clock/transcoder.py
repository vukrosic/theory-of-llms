"""A transcoder (the tool behind attribution graphs): sparse latents read the MLP's input and write the MLP's output.
usage: P=47 python3 transcoder.py MODEL.pt N K SEED [STEPS]   -> sae_runs/<model>_tc_<N>_<K>_<seed>.pt"""
import os, sys, time, torch
import mod, sae
P = mod.P
path, n, k, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
m = mod.Model(); m.load_state_dict(torch.load(path)); A, x, y = sae.acts(m)
t0 = time.time(); We, be, Wd, bd, bi = sae.train(A["z"], n, k, seed, steps, Y=A["out"])
Z = sae.encode(A["z"], We, be, bi, k); R = Z @ Wd.T + bd; Y = A["out"]
acc = lambda r: (sae.head(m, A["r1"] + r).argmax(-1) == y).float().mean().item()
kp = sae.kept(Y, R)
g = torch.zeros(P, n).index_add_(0, y, Z) / P; alive = (Z > 0).any(0)
sup = (g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) * alive; L = sae.label(Z); c2 = (L["share2d"] > 0.5) & alive
print(f"transcoder n={n} k={k} seed={seed} {time.time() - t0:.1f}s  unexplained var {((R - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum()):.3f}  "
      f"acc with it in place of the MLP {acc(R):.3f} (model {acc(Y):.3f})")
print(f"  rebuilt MLP output: a alone + b alone {kp[0]:.2f}, the answer (a+b) {kp[1]:.2f}, the rest {kp[2]:.2f}")
print(f"  latents: alive {alive.sum().item()}, one answer only {(sup == 1).sum().item()}, a function of a+b (>80%) {((L['type'] == 2) & (L['share'] > 0.8) & alive).sum().item()}, "
      f"of a alone or b alone {((L['type'] < 2) & (L['share'] > 0.8) & alive).sum().item()}, >50% on one clock {c2.sum().item()}; answers per latent {torch.bincount(sup[alive], minlength=6)[1:6].tolist()} (1..5)")
os.makedirs("sae_runs", exist_ok=True)
torch.save(dict(We=We, be=be, Wd=Wd, bd=bd, bi=bi, k=k, labels=L), f"sae_runs/{os.path.basename(path)[:-3]}_tc_{n}_{k}_{seed}.pt")
import json
with open("tc_runs.jsonl", "a") as fh:
    fh.write(json.dumps(dict(model=os.path.basename(path), n=n, k=k, seed=seed, acc=round(acc(R), 4), model_acc=round(acc(Y), 4), kept=kp)) + "\n")
