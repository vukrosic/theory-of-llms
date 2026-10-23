#!/bin/bash
# usage: run.sh TASKS "lrs" [extra env as VAR=val ...]
t=$1; lrs=$2; shift 2
env "$@" TASKS="$t" nice -n 20 python3 attn.py $lrs ${SEED:-0} 2>&1 | sed -E 's/SOFTCAP=[0-9.]+ GATE=[0-9]+ //; s/D=16 steps=800 mom=0.95 //'
