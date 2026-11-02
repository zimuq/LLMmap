#!/bin/bash
# D035 duplicate-submission watcher: $1 $2 = job ids of one step; once one is RUNNING, cancel the other; wait for it to end.
a=$1; b=$2
while :; do
  sa=$(squeue -h -j $a -o %T 2>/dev/null); sb=$(squeue -h -j $b -o %T 2>/dev/null)
  if [ "$sa" = RUNNING ]; then scancel $b; r=$a; break; fi
  if [ "$sb" = RUNNING ]; then scancel $a; r=$b; break; fi
  [ -z "$sa" ] && [ -z "$sb" ] && { echo "both gone"; exit 1; }
  sleep 20
done
echo "running $r $(date +%T)"
while [ -n "$(squeue -h -j $r 2>/dev/null)" ]; do sleep 30; done
sacct -j $r -X -n -o State,Elapsed
log=$(ls results/D035/logs/*-$r.out); grep -E "^\[A\] \[" $log; tail -1 $log
