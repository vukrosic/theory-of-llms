import json, math, numpy as np
R=[json.loads(l) for l in open('attn.jsonl')]
for r in R: r.setdefault('wd',0.0); r.setdefault('B',8)
def sel(qk,kw): 
    d={r['lr']:r for r in R if r['qknorm']==qk and all(r[a]==b for a,b in kw.items())}; return [d[k] for k in sorted(d)]
def best(rows):                     # parabola in log lr through the 3 points around min
    L=[r['loss'] for r in rows]; x=[math.log(r['lr']) for r in rows]; i=min(max(int(np.argmin(L)),1),len(L)-2)
    p=np.polyfit(x[i-1:i+2],L[i-1:i+2],2); return math.exp(-p[1]/(2*p[0])) if p[0]>0 else rows[i]['lr']
def star(rows):
    thr=5*next(r['maxlogit_peak'] for r in rows if r['lr']==0.014)
    for a,b in zip(rows,rows[1:]):
        if a['maxlogit_peak']<thr<=b['maxlogit_peak']:
            t=(math.log(thr)-math.log(a['maxlogit_peak']))/(math.log(b['maxlogit_peak'])-math.log(a['maxlogit_peak'])); return math.exp(math.log(a['lr'])+t*(math.log(b['lr'])-math.log(a['lr'])))
base=dict(B=8,steps=800,D=16,mom=0.95,wd=0.0)
for name,kw in [("B4 s1600",dict(B=4,steps=1600)),("B8 s800",{}),("B32 s200",dict(B=32,steps=200)),("mom0.9",dict(mom=0.9)),("mom0.98",dict(mom=0.98)),("D32",dict(D=32)),("D64",dict(D=64)),("s300",dict(steps=300)),("s1200",dict(steps=1200))]:
    k={**base,**kw}; n=sel(0,k); q=sel(1,k); s=star(n); bn=best(n); bq=best(q) if len(q)>=4 else None
    print(f"{name:9} lr*={s:.3f} best(noQK)={bn:.4f} best(QK)={('%.4f'%bq) if bq else 'n/a'} lr*/best(noQK)={s/bn:.1f} lr*/best(QK)={('%.1f'%(s/bq)) if bq else 'n/a'}")
w=sel(0,{**base,'wd':0.1}); print("WD0.1 lr*=%.3f best=%.4f"%(star(w),best(w)))
