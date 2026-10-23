#!/bin/bash
# usage: dil.sh TASK "ns" "lrs" QKNORM [env...]  -> TASK + first n CHAIN types
task=$1; ns=$2; lrs=$3; shift 3
for n in $ns; do
  s=$(python3 -c "
C='base s0 s2 beta0.5 beta8 p0.1 p0.5 d0.25 d0.9 c0.5 c1.0 mix0 mix1 mix2 mix3 mix4'.split()
print('$task,'+','.join(C[:$n]))")
  echo "n=$n"; ./run.sh "$s" "$lrs" "$@"
done
