import json, collections, sys
f = dict(kv.split("=") for kv in sys.argv[1:])   # filters, e.g. K=8 BURST=6 steps=300
R = collections.defaultdict(list)
for l in open("runs.jsonl"):
    r = json.loads(l)
    if all(str(r.get(k)) == v for k, v in f.items()): R[(r["D"], r["L"], r["T"], r["steps"], r["lr"])].append((r["seed"], r["recog"], r["copy"]))
for k in sorted(R):
    v = list({s: (a, b) for s, a, b in R[k]}.values())
    print("D=%d L=%d T=%d steps=%d lr=%g" % k, len(v), "recog %.2f copy %.2f" % (sum(a for a, b in v) / len(v), sum(b for a, b in v) / len(v)), [round(b, 2) for a, b in v])
