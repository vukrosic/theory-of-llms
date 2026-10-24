"""Which part of the activations does an SAE keep? Splits X (all P*P pairs at '=') into
f(a) + g(b) (each number alone), q((a+b) mod P) (the answer), and the rest, and reports the share of each part the SAE rebuilds.
usage: P=47 python3 parts.py MODEL.pt SAE.pt SITE"""
import sys, torch
import mod, sae
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); S = torch.load(sys.argv[2]); A, x, y = sae.acts(m); site = sys.argv[3]
X = A[site]; Z = sae.encode(X, S["We"], S["be"], S["bd"], S["k"]); R = Z @ S["Wd"].T + S["bd"]
split = sae.split
px, pr = split(X), split(R)
for name, a_, b_ in zip(["a alone + b alone", "the answer (a+b)", "the rest"], px, pr):
    print(f"{name:18}: {a_.pow(2).sum() / sum(p.pow(2).sum() for p in px):.2f} of the variance; SAE rebuilds {1 - (a_ - b_).pow(2).sum() / a_.pow(2).sum():+.2f} of it")
if site == "mlp":                                          # the part that reaches the answer scores: push each part through down
    for name, a_ in zip(["a alone + b alone", "the answer (a+b)", "the rest"], px):
        o = m.blk.down(a_.reshape(P * P, -1)); print(f"   after the down matrix, {name:18}: {o.pow(2).sum() / sum(m.blk.down(p.reshape(P*P,-1)).pow(2).sum() for p in px):.2f}")
