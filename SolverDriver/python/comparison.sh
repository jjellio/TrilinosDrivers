#!/bin/bash

CSVDIR=/home/www/datasets/onyx-v-cori

BL="onyx-frozen onyx"

COMPS="reboot switches"

for data in ${BL}; do
  for comp in ${COMPS}; do
    for kernel in tpetra muelu; do
      d="${data}_v_cori-${kernel}-${comp}"
      mkdir ${d}
      pushd ${d}
      CMD="${HOME}/src/TrilinosDrivers/SolverDriver/python/plot_openmp_weak.py"
      OPTS=""
      read -r -d '' OPTS <<- EOM
           --dataset=${CSVDIR}/${data}-${kernel}.csv
           --baseline=${CSVDIR}/cori-${kernel}-${comp}.csv
           --scaling=weak
           --max_only
           --max_nodes=64
           --min_procs_per_node=4
           --ymin="bl_speedup=0.5"
           --ymax="bl_speedup=1.6"
           --annotate_filenames
           --plot=raw_data,bl_speedup
           --plot_titles=ht
EOM

      if [ "$kernel" == "muelu" ]; then
        OPTS="${OPTS} --study=muelu_constructor --restrict_timer_labels=muelu_levels --sort_timer_labels=maxT"
      elif [ "$kernel" == "tpetra" ]; then
        OPTS="${OPTS} --study=linearAlg"
      fi
      
      echo "$CMD $OPTS"
      ls -1 ${CSVDIR}/${data}-${kernel}.csv
      ls -1 ${CSVDIR}/cori-${kernel}-${comp}.csv
      echo $CMD $OPTS &> gen.log
      $CMD $OPTS &> gen.log &

      popd
    done
  done
done
