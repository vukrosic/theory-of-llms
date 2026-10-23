import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'runs.jsonl')]
d=collections.defaultdict(dict)
for r in rows:
    if r['qkclip']==0: d[(r['B'],r['steps'],r['qknorm'])][r['lr']]=r
for (B,S) in sorted({(k[0],k[1]) for k in d}):
    n,q=d[(B,S,0)],d[(B,S,1)]; lrs=sorted(set(n)&set(q))
    if not lrs: continue
    best=min(lrs,key=lambda l:q[l]['loss']); bestn=min(lrs,key=lambda l:n[l]['loss']); p0=n[best]['maxlogit_peak']
    thr=next((l for l in lrs if l>=best and n[l]['maxlogit_peak']>=5*p0),None)
    print(f"B{B} steps{S}: best lr QKN {best} (noQKN {bestn}), peak@best {p0:.1f}, blow-up lr {thr}, ratio {thr/best if thr else float('nan'):.1f}, loss gap at best {n[best]['loss']-q[best]['loss']:+.3f}")
    print("   "+" ".join(f"{l}:{n[l]['maxlogit_peak']:.0f}/{n[l]['loss']-q[l]['loss']:+.3f}/{q[l]['loss']:.3f}" for l in lrs))
