#!/bin/bash
# D038 duplicate-submission watcher (gg + gh-dev + gh): keep the first job id to reach RUNNING, cancel the rest, wait
# for it. Prints job STATE ONLY, never log lines.
ids="$@"
while :; do
  r=""; alive=0
  for j in $ids; do s=$(squeue -h -j $j -o %T 2>/dev/null); [ -n "$s" ] && alive=1; [ "$s" = RUNNING ] && { r=$j; break; }; done
  [ -n "$r" ] && break
  [ $alive = 0 ] && { echo "all gone"; exit 1; }
  sleep 20
done
for j in $ids; do [ $j != $r ] && scancel $j; done
echo "running $r $(date +%T)"
while [ -n "$(squeue -h -j $r 2>/dev/null)" ]; do sleep 30; done
sacct -j $r -X -n -o JobID,Partition,State,Elapsed,ExitCode
