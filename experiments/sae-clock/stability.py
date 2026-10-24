"""Do SAEs with different seeds find the same latents? For each latent: best |cosine| between its decoder vector and any
decoder vector of the other SAE. Split into latents that fire on one answer and latents that fire on several.
usage: P=47 python3 stability.py MODEL.pt SITE SAE_A.pt SAE_B.pt [...]"""
import itertools, sys, torch
import mod, sae
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); A, x, y = sae.acts(m); X = A[sys.argv[2]]; s = (x[:, 0] + x[:, 1]) % P
info = []
for f in sys.argv[3:]:
    S = torch.load(f); Z = sae.encode(X, S["We"], S["be"], S["bd"], S["k"]); g = torch.zeros(P, Z.shape[1]).index_add_(0, s, Z) / P
    alive = (Z > 0).any(0); one = ((g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) == 1) & alive
    info.append((torch.nn.functional.normalize(S["Wd"], dim=0), alive, one, g.argmax(0)))
res = {"one": [], "multi": []}; same_ans = []; meaning = []; cos_same = []
for (Da, aa, oa, ans_a), (Db, ab, ob, ans_b) in itertools.permutations(info, 2):
    C = (Da.T @ Db).abs(); C[:, ~ab] = 0; best, j = C.max(1)
    res["one"] += best[oa].tolist(); res["multi"] += best[aa & ~oa].tolist()
    both = oa & ob[j] & (best > 0.9); same_ans += (ans_a[both] == ans_b[j[both]]).float().tolist()
    for i in oa.nonzero()[:, 0].tolist():                  # same answer coded by a one-answer latent in the other SAE?
        js = (ob & (ans_b == ans_a[i])).nonzero()[:, 0]
        meaning.append(len(js) > 0); 
        if len(js): cos_same.append((Da[:, i] @ Db[:, js]).abs().max().item())
for kname, v in res.items():
    v = torch.tensor(v); print(f"{kname:5}-answer latents: {len(v)}, found again in another seed (cos > 0.9): {(v > 0.9).float().mean():.2f}, median best cos {v.median():.2f}")
print(f"  matched one-answer pairs that code the same answer: {torch.tensor(same_ans).mean():.2f}")
print(f"  one-answer latents whose answer also has a one-answer latent in the other seed: {torch.tensor(meaning).float().mean():.2f};"
      f" cosine between those two latents: median {torch.tensor(cos_same).median():.2f}, > 0.9: {(torch.tensor(cos_same) > 0.9).float().mean():.2f}")
