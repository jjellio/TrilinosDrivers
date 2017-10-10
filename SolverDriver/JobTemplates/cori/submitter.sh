#!/bin/bash

#switch wait
switch_wait="72:00:00"

#for n in 1 2 8 16 64 128 256 512 1024; do
for n in 1 2 4 8 16 32 64 128 256 512 1024; do
#for n in 1; do
  num_switches=$(( $n / 384 + 1))
  o=$(sbatch  --nodes=${n} --switches=${num_switches}@${switch_wait}  ./fused-submission.sh)

  j=`echo $o | cut -f4 -d' '`
  echo ${j}
done

