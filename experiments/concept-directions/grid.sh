#!/bin/zsh
# usage: grid.sh "W STEPS SEED" ...   runs 2 at a time, waits while 8+ heavy python jobs run
one() {
  while [ $(ps aux | awk '/[p]ython3/ && $3>50' | wc -l) -ge 7 ]; do sleep 3; done
  set -- ${=1}
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DRIFT=1 WIDTH=$1 MIX=0 MOE=8 MSIGN=ns EMB_MULT=8 nice -n 20 python3 dirs.py $3 $2 > logs/w$1_s$2_seed$3.txt 2>&1
  head -1 logs/w$1_s$2_seed$3.txt
}
mkdir -p logs
for a in "$@"; do one "$a" & sleep 1; while [ $(jobs | wc -l) -ge 2 ]; do sleep 1; done; done; wait
