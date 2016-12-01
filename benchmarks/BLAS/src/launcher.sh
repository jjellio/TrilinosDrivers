#!/bin/bash
CORES=64
MAX_HTs=4


BLASLIB=libsci
THREADING=intel
ARCH=knl

EXE_ID=${BLASLIB}_${THREADING}_${ARCH}
EXE=./BlasTests.${EXE_ID}

AGG_FILE=aggregate.csv
DETAIL_FILE=details.csv

for HTs in $(seq 1 ${MAX_HTs}); do
  CSVFILE=${EXE}-HT_${HTs}.csv
  AGG_CSVFILE=${EXE}-HT_${HTs}_agg.csv
  DETAIL_CSVFILE=${EXE}-HT_${HTs}_details.csv
  OLOGFILE=${EXE}-HT_${HTs}.out
  ELOGFILE=${EXE}-HT_${HTs}.err
  
  rm -f ${AGG_CSVFILE} ${DETAIL_CSVFILE}

  NP=${CORES}
  while [  $NP -gt 0 ]; do
    echo NP=${NP}
    echo HTs=${HTs}
  
    let n=${NP}
    let j=${HTs}
    let d=$((${CORES}*${HTs}/${NP}))
  
    for policy in "active" "passive"; do
    
      for places in "cores" "threads"; do
        rm -f ${AGG_FILE} ${DETAIL_FILE}

        echo aprun -n ${n} -j ${j} -d ${d} -cc depth -e OMP_NUM_THREADS=${d} -e OMP_WAIT_POLICY=${policy} -e OMP_PLACES=${places} -e OMP_DISPLAY_ENV=verbose -e KMP_AFFINITY=verbose ${EXE}
             aprun -n ${n} -j ${j} -d ${d} -cc depth -e OMP_NUM_THREADS=${d} -e OMP_WAIT_POLICY=${policy} -e OMP_PLACES=${places} -e OMP_DISPLAY_ENV=verbose -e KMP_AFFINITY=verbose ${EXE}  2>> ${ELOGFILE} 1>> ${OLOGFILE}
        cat ${AGG_FILE} >> ${AGG_CSVFILE}
        cat ${DETAIL_FILE} >> ${DETAIL_CSVFILE}

      done
    done

    let NP=NP/2  
  done

  head -n1 ${AGG_CSVFILE} > tmp.csv
  cat ${AGG_CSVFILE} | grep -v "Label" >> tmp.csv
  mv tmp.csv ${AGG_CSVFILE}

  head -n1 ${DETAIL_CSVFILE} > tmp.csv
  cat ${DETAIL_CSVFILE} | grep -v "Label" >> tmp.csv
  mv tmp.csv ${DETAIL_CSVFILE}

  
  cat ${OLOGFILE} | grep "Label" | head -n1 > ${CSVFILE}
  cat ${OLOGFILE} | grep -v "Label" | grep -v "utime" >> ${CSVFILE}
done

#aprun -n2 -j1 -d 16 -cc depth -e OMP_WAIT_POLICY=active -e OMP_PLACES=cores -e OMP_DISPLAY_ENV=verbose ./BlasTests.mkl_intel_hsw
