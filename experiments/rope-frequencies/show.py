"""summarize a runs file: mean over seeds per config. usage: python3 show.py FILE [keys...]"""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
cfgk = ("width", "layers", "heads", "T", "steps", "base", "rot", "span", "lr", "qknorm", "ext_method", "ext_steps")
g = collections.defaultdict(list)
for r in rows: g[tuple(r.get(k) for k in cfgk)].append(r)
cols = sys.argv[2:] or ["text.1x_none", "text.4x_none", "text.4x_ntk", "text.4x_pi", "text.4x_yarn", "near.1x_none", "far.4x_none", "far.4x_ntk", "drop.fast", "drop.slow"]
def get(r, c):
    a, b = c.split("."); v = r[a][b]; return v[0] if isinstance(v, list) else v
print(" ".join(f"{k}" for k in cfgk), "n", *cols)
for k, rs in g.items():
    vals = []
    for c in cols:
        xs = [get(r, c) for r in rs if c.split(".")[1] in r[c.split(".")[0]]]
        m = sum(xs) / len(xs) if xs else float("nan"); sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        vals.append(f"{m:.3f}±{sd:.3f}")
    print(*k, len(rs), *vals)
