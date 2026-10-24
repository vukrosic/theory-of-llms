#!/bin/zsh
# usage: ./run.sh OUTFILE "ENV=.. ENV=.." seed [seed...]   runs seeds 2 at a time, waits while 6+ heavy python jobs run
out=$1; cfg=$2; shift 2
for s in "$@"; do
  while [ $(ps -eo pcpu,command | awk '$1>40' | grep -c python) -ge 6 ]; do sleep 2; done
  env ${=cfg} OUT=$out OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 20 python3 rope.py $s > /dev/null &
  if (( $s % 2 == 1 )); then wait; fi
done
wait
