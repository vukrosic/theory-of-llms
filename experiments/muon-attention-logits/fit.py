import json, math, numpy as np
R=[json.loads(l) for l in open('attn.jsonl')]
R=[r for r in R if r['seed']==0]
key=lambda r:(r['D'],r['steps'],r['mom'])
cfgs=sorted({key(r) for r in R})
def lrstar(rows,thr=30):                      # lr where peak logit crosses thr (log-log interpolation)
    rows=sorted(rows,key=lambda r:r['lr'])
    for a,b in zip(rows,rows[1:]):
        if a['maxlogit_peak']<thr<=b['maxlogit_peak']:
            t=(math.log(thr)-math.log(a['maxlogit_peak']))/(math.log(b['maxlogit_peak'])-math.log(a['maxlogit_peak']))
            return math.exp(math.log(a['lr'])+t*(math.log(b['lr'])-math.log(a['lr'])))
print("config(D,steps,mom) | lr* (peak logit = 5x its value at lr .014) | best lr/loss noQK | best lr/loss QK | loss gap noQK-QK at lr 0.057 / 0.113")
for c in cfgs:
    n=[r for r in R if key(r)==c and r['qknorm']==0]; q=[r for r in R if key(r)==c and r['qknorm']==1]
    ls=lrstar(n,5*next(r['maxlogit_peak'] for r in n if r['lr']==0.014)); bn=min(n,key=lambda r:r['loss']); bq=min(q,key=lambda r:r['loss'])
    g=lambda lr:(next(r['loss'] for r in n if r['lr']==lr)-next(r['loss'] for r in q if r['lr']==lr)) if any(r['lr']==lr for r in q) else float('nan')
    print(c, f"{ls:.3f}" if ls else 'n/a', f"{bn['lr']}/{bn['loss']:.3f}", f"{bq['lr']}/{bq['loss']:.3f}", f"{g(0.057):+.3f}/{g(0.113):+.3f}")
# power-law fit above threshold: log L = a + b log lr + c log steps (D=16, mom .95, noQK, lr>=0.057)
X=[r for r in R if r['qknorm']==0 and r['D']==16 and r['mom']==0.95 and r['lr']>=0.057]
A=np.array([[1,math.log(r['lr']),math.log(r['steps'])] for r in X]); y=np.log([r['maxlogit_peak'] for r in X])
co=np.linalg.lstsq(A,y,rcond=None)[0]; print("above threshold (D16,mom.95): peak logit ~ lr^%.2f * steps^%.2f  (n=%d, resid sd %.2f in ln)"%(co[1],co[2],len(X),np.std(y-A@co)))
X=[r for r in R if r['qknorm']==0 and r['D']==16 and r['mom']==0.95 and r['lr']<=0.028 and r['steps']==800]
print("below threshold D16 steps800 peak logit:",[(r['lr'],r['maxlogit_peak']) for r in sorted(X,key=lambda r:r['lr'])])
X=[r for r in R if r['qknorm']==0 and r['mom']==0.95 and r['steps']==800 and r['lr']==0.014]; print("lr .014 by width:",[(r['D'],r['maxlogit_peak']) for r in X])
X=[r for r in R if r['qknorm']==1]; print("QK-norm peak logit max over all runs:",max(r['maxlogit_peak'] for r in X), "by width:", {d:max(r['maxlogit_peak'] for r in X if r['D']==d) for d in (16,32,64)})
# momentum: lr* vs 1-mom
ms=[(c[2],lrstar([r for r in R if key(r)==c and r['qknorm']==0],5*next(r['maxlogit_peak'] for r in R if key(r)==c and r['qknorm']==0 and r['lr']==0.014))) for c in cfgs if c[0]==16 and c[1]==800]; print("lr* vs mom:",ms)
