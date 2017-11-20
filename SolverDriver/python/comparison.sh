#!/bin/bash

CSVDIR=${HOME}/perf-data/datasets/onyx-v-cori

BL="onyx-frozen"
machine=theta
expected_baseline_speedup=1.0
COMPS="default reboot switches"

for machine in theta cori; do
if [ "$machine" == "theta" ]; then
expected_baseline_speedup=1.0;
else
expected_baseline_speedup=`echo "scale=10; 1.3/1.4" | bc`
fi


for data in ${BL}; do
  for comp in ${COMPS}; do
    for kernel in muelu tpetra; do
      d="${data}_v_${machine}-${kernel}-${comp}"
      mkdir ${d}
      pushd ${d}
      CMD="${HOME}/src/TrilinosDrivers/SolverDriver/python/plot_openmp_weak.py"
      OPTS=""
      read -r -d '' OPTS <<- EOM
           --dataset=${CSVDIR}/${data}-${kernel}.csv
           --baseline=${CSVDIR}/${machine}-${kernel}-${comp}.csv
           --expected_baseline_speedup=${expected_baseline_speedup}
           --scaling=weak
           --max_only
           --max_nodes=64
           --min_procs_per_node=4
           --ymin="bl_speedup=0.5"
           --ymax="bl_speedup=1.6"
           --annotate_filenames
           --plot=raw_data,bl_speedup
           --plot_titles=ht
           --number_plots=false
EOM

      if [ "$kernel" == "muelu" ]; then
        OPTS="${OPTS} --study=muelu_constructor --restrict_timer_labels=muelu_levels --sort_timer_labels=maxT"
        OPTS+=" --average=ns"
      elif [ "$kernel" == "tpetra" ]; then
        OPTS="${OPTS} --study=linearAlg --sort_timer_labels=maxT"
        OPTS+=" --average=cc"
      fi

      echo "$CMD $OPTS"
      ls -1 ${CSVDIR}/${data}-${kernel}.csv
      ls -1 ${CSVDIR}/${machine}-${kernel}-${comp}.csv
      echo $CMD $OPTS &> gen.log
      $CMD $OPTS >> gen.log 2>&1 &

      popd
    done
  done
done
done
