#!/bin/bash
# D037 duplicate-submission watcher: $1 $2 = job ids of one step; once one is RUNNING, cancel the other; wait for it
# to end. Prints job STATE ONLY -- never log lines (D036 Disclosure 1; D037 S3: no label before the tail plot).
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
sacct -j $r -X -n -o JobID,State,Elapsed,ExitCode
