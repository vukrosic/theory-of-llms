import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'runs.jsonl')]
key=lambda r:(r['D'],r['H'],r['nl'],r['mom'],r.get('B'),r['qknorm'],r['qkclip'],r['lr'])
d={}
for r in rows: d[key(r)]=r
shapes=sorted({(k[0],k[1],k[2],k[3],k[4]) for k in d})
for s in shapes:
    print(f"D{s[0]} H{s[1]} hd{s[0]//s[1]} NL{s[2]} mom{s[3]} B{s[4]}")
    for (qn,c) in sorted({(k[5],k[6]) for k in d if k[:5]==s}):
        out=[]
        for lr in (0.014,0.057,0.113):
            r=d.get(s+(qn,c,lr))
            out.append(f"{r['loss']:.3f}/{r['maxlogit_peak']:.0f}/nc{r['nclip']}" if r else "   -   ")
        print(f"   qknorm={qn} clip={c:g}: "+"  ".join(out))
