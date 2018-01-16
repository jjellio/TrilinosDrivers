#!/bin/bash

JID=$1
SORT_DIR=/tmp/sparc-knl-cache/algo-sort
OUT_DIR=${SORT_DIR}/csv

pushd $JID;

echo JobID: $JID

# sparc.out.ats1-knl-001x64x1x4-quad_flat_hbm.txt
sparc_out=`ls -1 sparc.out.*.txt 2>/dev/null`

have_timer_file="false"

if [ -f "teuchos_timer_output.txt" ]; then
  have_timer_file="true"
fi

if [ -f "${sparc_out}" ]; then
  echo "Sparc.out: $sparc_out"
elif [ -f "outputs.tar.xz" ]; then
  mkdir -p tmp
  cd tmp
  tar xf ../outputs.tar.xz
  sparc_out=`ls -1 sparc.out.*.txt 2>/dev/null`
  if [ -f "${sparc_out}" ]; then
    echo "Sparc.out: $sparc_out"
  else
    echo "extracted to tmp folder, but failed ot find sparc.out"
    ls -l
    cd ..
    rm -rf tmp
    echo "$JID Failed"
    exit;
  fi

  if [ "$have_timer_file" == "false" ] && [ -f "teuchos_timer_output.txt" ]; then
    have_timer_file="true"
    cp -a teuchos_timer_output.txt ..
  else
    cd ..
    rm -rf tmp
    echo "$JID Failed no timer file"
    exit;
  fi

else
  echo "No outputs.tar.xz!"
  ls -l
  echo "$JID Failed"
  exit;
fi

echo Processing $sparc_out;

numactl_=`grep -Po -- 'numactl\s+-m\s*\d+' ${sparc_out} | grep -Po -- '\d+'`;
freq_=`grep -Po -- '--cpu-freq=\w+' ${sparc_out} | cut -f2 -d '='`;

if [ "$numactl_" == "0" ]; then
  numactl_="cache"
elif [ "$numactl_" == "1" ]; then
  numactl_="flat_hbm"
else
  echo "$numactl_ :Failed, could not parse"
  exit;
fi

# sparc.out.ats1-knl-001x64x1x4-quad_flat_hbm.txt
decomp_=`echo ${sparc_out} | tr '-' '\n' | grep -Po -- '\d+x\d+x\d+x\d+'`

csv_file="${OUT_DIR}/${decomp_}_${numactl_}_${freq_}.csv"
header_file="${OUT_DIR}/header.csv"

if [ `basename $PWD` == "tmp" ]; then
  cd ..
  rm -rf tmp
fi

if [ ! -f "$header_file" ]; then
  head -n1 teuchos_timer_output.txt.csv > ${header_file}
fi

rm -f teuchos_timer_output.txt*.csv
~/src/TrilinosDrivers/SolverDriver/python/teuchos_log_to_csvs.py --log=teuchos_timer_output.txt
tail -n +2  teuchos_timer_output.txt.csv >> ${csv_file}
rm -f teuchos_timer_output.txt*.csv


popd

#cp -ra $JID ${my_dir}


