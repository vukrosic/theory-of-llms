"""Baseline: label the model's own 128 MLP neurons (and the 64 residual dims) the same way as SAE latents.
usage: P=47 python3 neurons.py MODEL.pt"""
import sys, torch
import mod, sae
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); A, x, y = sae.acts(m)
for site in ("mlp", "res"):
    L = sae.label(A[site]); c2 = L["share2d"] > 0.5; bc = torch.bincount(L["clock2d"][c2], minlength=mod.P // 2 + 1)
    print(f"{site}: units with >50% of their (a, b) map on one clock: {c2.sum().item()} of {len(c2)}; median share {L['share2d'].median():.2f}; clocks "
          + " ".join(f"{f}:{bc[f].item()}" for f in bc.argsort(descending=True)[:8].tolist() if bc[f] > 0))
