import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'runs.jsonl')]
S=lambda r:(r['D'],r['H'],r['nl'],r['mom'],r['B'])
d={}
for r in rows: d[S(r)+(r['qknorm'],r['qkclip'],r['lr'])]=r
taus=[0,100,30,20,15,10,7,5,3.5,2.5]
for s in sorted({S(r) for r in rows}):
    hl=d.get(s+(0,0,0.014)); q=d.get(s+(1,0,0.113))
    if not hl or not q: continue
    med=st.median([v for l in hl['heads'] for v in l])
    print(f"D{s[0]} H{s[1]} NL{s[2]} mom{s[3]} B{s[4]}: healthy max {hl['maxlogit_peak']:.0f} median head {med:.1f} QKnorm@.113 {q['loss']:.3f}; tau: gap@.113 [cost@.014]")
    out=[]
    for t in taus:
        a=d.get(s+(0,t,0.113)); c=d.get(s+(0,t,0.014))
        if a and c: out.append(f"{t:g}({t/med:.2f}x): {a['loss']-q['loss']:+.3f}[{c['loss']-hl['loss']:+.3f}]")
    print("   "+"  ".join(out))
