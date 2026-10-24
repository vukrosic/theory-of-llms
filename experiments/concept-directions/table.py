"""summary of runs.jsonl: mean over seeds per (width, steps)"""
import json, collections, sys
R = [json.loads(l) for l in open("runs.jsonl") if '"drift"' in l and json.loads(l).get("mix") == 0]
R = list({(r["width"], r["steps"], r["seed"]): r for r in R}.values())   # one row per run (the latest)
grp = collections.defaultdict(list)
for r in R: grp[(r["width"], r["steps"])].append(r)
avg = lambda v: sum(v) / len(v)
mean_d = lambda d: avg(list(d.values())) if d else float("nan")
print("width steps n | count acc base | site: lin turn pc1 chord1 chordB line1 lineB first1 rand  || rel base: site best  || drift(half): count probe/chord/cos rel probe/best")
for (w, st), rs in sorted(grp.items()):
    out = f"{w:4d} {st:5d} {len(rs)} | {avg([r['acc']['count'] for r in rs]):.2f} {avg([r['concepts']['count']['base_acc'] for r in rs]):.2f} |"
    for site in ("1", "2"):
        S = [r["concepts"]["count"]["sites"][site] for r in rs]
        out += f" s{site}: {avg([s['lin'] for s in S]):.2f} {avg([s['turn'] for s in S]):.2f} {avg([s['pc1'] for s in S]):.2f} " + " ".join(
            f"{avg([mean_d(s['steer'][m][k]) for s in S]):.2f}" for m, k in (("chord", "s1"), ("chord", "best"), ("line", "s1"), ("line", "best"), ("first", "s1"), ("rand", "best")))
    out += f" || {avg([r['concepts']['rel']['base_acc'] for r in rs]):.2f}"
    for site in ("1", "2"):
        out += f" s{site} {avg([avg([v['best'] for v in r['concepts']['rel']['sites'][site]['steer'].values()]) for r in rs]):.2f}"
    half = lambda r: r["drift"][str(int(0.5 * r["steps"]))]
    for site in ("1", "2"):
        out += f" || s{site} {avg([half(r)[site]['count']['probe'] for r in rs]):.2f}/{avg([half(r)[site]['count']['chord_best'] for r in rs]):.2f}/{avg([half(r)[site]['count']['cos'] for r in rs]):.2f} {avg([half(r)[site]['rel']['probe'] for r in rs]):.2f}/{avg([half(r)[site]['rel']['best'] for r in rs]):.2f}"
    print(out)
