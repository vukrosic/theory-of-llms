"""What does Muon's best lr depend on? One factor at a time around the ladder.py toy.
usage: B=8 D=16 STEPS=800 [NL= T= MOM= WD= WARM= DFRAC= MIX=] python3 factors.py SEED  -> factors.jsonl (15 lrs, full config per row)
       B=8 D=16 STEPS=800 [...] python3 factors.py time                               -> seconds for one run at lr 0.014
       python3 factors.py fit                                                         -> table: factor | value | best lr | band | moves?
MIX: mix20 (default, endpoint.py's 20 types), copy (chain c1.0), reverse, sort, chain (base chain), text (bytes of
$CORPUS (any plain-text file, default corpus.txt next to this script): non-overlapping T+1 windows from the first 95%, eval 512 windows from the last 5%).
"""
import sys, os, json, math, time, torch
import endpoint as E
import scale as S
import ladder as Ld
import mixprobe as M
L = E.L; HERE = L.HERE; MIX = os.environ.get("MIX", "mix20")
TYPES = dict(mix20=E.TYPES, copy=["c1.0"], reverse=["reverse"], sort=["sort"], chain=["base"])
CFG = dict(B=L.B, D=L.D, NL=L.NL, T=L.T, MOM=S.MOM, WD=S.WD, WARM=S.WARM, DFRAC=S.DFRAC if S.DECAY else 0.0, MIX=MIX, steps=E.STEPS)
BASE = dict(B=8, D=16, NL=2, T=32, MOM=0.95, WD=0.0, WARM=20, DFRAC=0.5, MIX="mix20", steps=int(os.environ.get("BASESTEPS", 800)))
LRS = [float(x) for x in os.environ["LRS"].split(",")] if "LRS" in os.environ else Ld.LRS   # knob: lr grid (default 15 lrs)
FOUT = f"{HERE}/{os.environ.get('FOUT', 'factors.jsonl')}"                                  # knob: output file


def text(steps):
    raw = torch.frombuffer(bytearray(open(os.environ.get("CORPUS", f"{HERE}/corpus.txt"), "rb").read()), dtype=torch.uint8).long()
    cut = int(0.95 * len(raw)); w = L.T + 1
    tr = raw[:cut - cut % w].view(-1, w); tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))][:steps * L.B]
    ev = raw[cut:cut + 512 * w].view(512, w); assert len(tr) == steps * L.B
    return tr, {"text": ev}, {"text": torch.ones(L.T, dtype=torch.bool)}


def data(): return text(E.STEPS) if MIX == "text" else M.data(TYPES[MIX], E.STEPS)


if __name__ == "__main__":
    if sys.argv[1] == "time":
        tr, ev, mask = data(); t0 = time.time(); l = S.train(tr, ev, mask, 0.014, 0, E.STEPS); print(f"{CFG}: {time.time() - t0:.2f}s loss {l:.4f}"); sys.exit()
    if sys.argv[1] != "fit":
        seed = int(sys.argv[1]); tr, ev, mask = data()
        for lr in LRS:
            t0 = time.time(); loss = S.train(tr, ev, mask, lr, seed, E.STEPS)
            with open(FOUT, "a") as fo: fo.write(json.dumps(dict(**CFG, lr=lr, seed=seed, loss=round(loss, 5))) + "\n")
            print(f"{CFG} lr {lr} seed {seed}: {loss:.4f} ({time.time() - t0:.1f}s)", flush=True)
        sys.exit()
    runs = {}
    for l in open(FOUT):
        r = json.loads(l); k = tuple(r[f] for f in BASE); runs.setdefault(k, {}).setdefault(r["lr"], []).append(r["loss"])
    print(f"{'factor':6} {'value':>7} {'steps':>5} {'best lr':>8} {'lrs within 0.005':>17} {'seeds':>5} {'min loss':>8}")
    for f in BASE:
        if f == "steps": continue
        rows = sorted((k, d) for k, d in runs.items() if all(k[j] == BASE[g] for j, g in enumerate(BASE) if g not in (f, "steps")))
        grp = (lambda k: k[0] * k[-1]) if f == "B" else (lambda k: k[-1])      # batch: grouped by tokens (B x steps)
        for st in sorted({grp(k) for k, _ in rows}):
            g = [(k, d) for k, d in rows if grp(k) == st]
            if len(g) < 2: continue
            for k, d in g:
                lo, hi = Ld.band(d); print(f"{f:6} {str(k[list(BASE).index(f)]):>7} {k[-1]:5} {M.best7(d):8.4f} {lo:>8}-{hi:<8} "
                                           f"{min(len(v) for v in d.values()):5} {min(sum(v) / len(v) for v in d.values()):8.4f}")
            b = [M.best7(d) for _, d in g]; print(f"{f:6} {'ratio':>7} {'':5} {max(b) / min(b):8.2f}")
