"""Best Muon lr vs training length over a 16x range of steps, with and without the lr decay (scale.train, 15 lrs 0.0012-0.16).
usage: B=8 D=16 STEPS=400 [DECAY=0] python3 ladder.py SEED [LR,LR]-> appends (B, D, steps, lr, seed, loss) to ladder.jsonl / ladder_nodecay.jsonl
       [DECAY=0] python3 ladder.py fit                       -> best lr per steps (parabola, scale.best), near-best band, slope vs steps
Same model, data, schedule and update as scale.py; own output files because batch B differs from scale.jsonl's 32.
"""
import sys, os, json, math, time
import endpoint as E
import scale as S
L = E.L; HERE = L.HERE; DECAY = S.DECAY
OUT = f"{HERE}/ladder.jsonl" if DECAY else f"{HERE}/ladder_nodecay.jsonl"
LRS = [0.0012, 0.0018] + S.LRS                                         # no-decay best sits near scale.py's lowest lr


def band(d, tol=0.005):                                                 # lrs whose mean loss is within tol of the best mean
    Ls = {x: sum(v) / len(v) for x, v in d.items()}; lo = min(Ls.values()); ok = [x for x in sorted(Ls) if Ls[x] <= lo + tol]
    return ok[0], ok[-1]


if __name__ == "__main__":
    if sys.argv[1] != "fit":
        seed = int(sys.argv[1]); tr, ev, mask = E.data()
        for lr in (LRS if len(sys.argv) < 3 else [float(x) for x in sys.argv[2].split(",")]):
            t0 = time.time(); loss = S.train(tr, ev, mask, lr, seed, E.STEPS)
            with open(OUT, "a") as fo: fo.write(json.dumps(dict(B=L.B, D=L.D, steps=E.STEPS, lr=lr, seed=seed, loss=round(loss, 5))) + "\n")
            print(f"B {L.B} D {L.D} steps {E.STEPS} decay {DECAY} lr {lr} seed {seed}: {loss:.4f} ({time.time() - t0:.1f}s)", flush=True)
        sys.exit()
    runs = {}
    for l in open(OUT): r = json.loads(l); runs.setdefault((r["B"], r["D"], r["steps"]), {}).setdefault(r["lr"], []).append(r["loss"])
    for BD in sorted({k[:2] for k in runs}):
        rows = []
        for st in sorted(k[2] for k in runs if k[:2] == BD):
            d = runs[(*BD, st)]; b, edge = S.best(d); lo, hi = band(d); n = min(len(v) for v in d.values())
            bs = [S.best({x: [v[j]] for x, v in d.items()})[0] for j in range(n)]          # per-seed best lr
            rows.append((st, b, edge, bs))
            print(f"B {BD[0]} D {BD[1]} steps {st:5}  best lr {b:.4f}{' (edge)' if edge else ''}  within 0.005 of min: {lo}-{hi}  "
                  f"per-seed best {min(bs):.4f}-{max(bs):.4f}  min loss {min(sum(v) / len(v) for v in d.values()):.4f}  seeds {n}")
        ok = [r for r in rows if not r[2]]
        if len(ok) > 1:
            xs = [math.log(r[0]) for r in ok]; n = min(len(r[3]) for r in ok)
            sl = [S.slope(xs, [math.log(r[3][j]) for r in ok]) for j in range(n)]; mu = sum(sl) / n
            se = (sum((s - mu) ** 2 for s in sl) / (n - 1) / n) ** 0.5 if n > 1 else float("nan")
            print(f"best lr ~ steps^{S.slope(xs, [math.log(r[1]) for r in ok]):+.3f} (seed-mean losses); "
                  f"per-seed slopes {' '.join(f'{s:+.2f}' for s in sl)} -> {mu:+.3f} +- {se:.3f} (se)")
