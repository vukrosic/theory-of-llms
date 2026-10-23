#!/bin/zsh
# usage: ./run.sh LR "label" [ENV=val ...]   runs seeds 0,1,2 (2 parallel, then 1), prints mean loss, mean peak, mean final, mean nclip, first
cd "$(dirname "$0")"; lr=$1; label=$2; shift 2
export FOUT=tmp_$$.jsonl; export QKNORM=${QKNORM:-0}
( env "$@" nice -n 20 python3 attn.py $lr 0 >/dev/null ) &
( env "$@" nice -n 20 python3 attn.py $lr 1 >/dev/null ) &
wait
env "$@" nice -n 20 python3 attn.py $lr 2 >/dev/null
python3 - "$label" $FOUT <<'PY'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[2])]; rows.sort(key=lambda r:r['seed'])
f=lambda k:st.mean(r[k] for r in rows)
out=f"{sys.argv[1]:42s} loss {f('loss'):.3f} [{' '.join('%.2f'%r['loss'] for r in rows)}]  peak {f('maxlogit_peak'):8.1f}  final {f('maxlogit_final'):7.1f}  nclip {f('nclip'):6.0f}  first {[r['first'] for r in rows]}"
print(out); open('summary.txt','a').write(out+'\n')
PY
rm -f $FOUT
