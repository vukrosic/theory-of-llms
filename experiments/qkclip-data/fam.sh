#!/bin/bash
# usage: fam.sh "lrs" env... ; loops over families
lrs=$1; shift
for f in "" c1.0 d0.9 beta8 p0.5 algo chains; do ./run.sh "$f" "$lrs" "$@"; done
