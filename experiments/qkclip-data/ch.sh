#!/bin/bash
# usage: ch.sh "n list" "lrs" env... -> first n CHAIN types only
ns=$1; lrs=$2; shift 2
for n in $ns; do
  s=$(python3 -c "
C='base s0 s2 beta0.5 beta8 p0.1 p0.5 d0.25 d0.9 c0.5 c1.0 mix0 mix1 mix2 mix3 mix4'.split()
print(','.join(C[:$n]))")
  echo "chains n=$n"; ./run.sh "$s" "$lrs" "$@"
done
