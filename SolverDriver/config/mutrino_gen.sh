#!/bin/bash


for job_type in hsw knl_flat knl_cache; do

ARCH=""
KNL_TYPE=""

if [ "${job_type}" == "hsw" ]; then
  ARCH=hsw
  KNL_TYPE=""
elif [ "${job_type}" == "knl_flat" ]; then
  ARCH=knl
  KNL_TYPE=quad_flat
elif [ "${job_type}" == "knl_cache" ]; then
  ARCH=knl
  KNL_TYPE=quad_cache
fi

HTs=()
PEs=()
PE_PER_NUMA=-1
NUM_NUMA=-1
BIN=""
ROOT_DIR=/home/jjellio/trilinos-perf
GOLD_DIR=${ROOT_DIR}/etc/gold
DRIVER_XML=${ROOT_DIR}/etc/driver.xml
FILE_PREFIX=""
MSUB_NODE_SUFFIX=""
AFFINITY_CHECK_EXE=""

if [ "${ARCH}" == "knl" ]; then
  HTs=(1 2 3 4)
  PEs=(1 2 4 8 16 32)
  NUM_NUMA=1
  # save 4 cores for the OS
  PE_PER_NUMA=64
  THREADS_PER_CORE=4
  FILE_PREFIX="${ROOT_DIR}/runs/knl_${KNL_TYPE}"
  SCALING_EXE="${ROOT_DIR}/bin/SolverDriver_knl_openmp.exe"
  AFFINITY_CHECK_EXE="${ROOT_DIR}/bin/omp_diag.knl"
  MSUB_NODE_SUFFIX="ppn=68:knl -los=CLE_quad_cache"

  if [ "${KNL_TYPE}" == "quad_flat" ]; then
     SCALING_EXE="numactl -p 1 ${SCALING_EXE}"
     AFFINITY_CHECK_EXE="numactl -p 1 ${AFFINITY_CHECK_EXE}"
     MSUB_NODE_SUFFIX="ppn=68:knl -los=CLE_quad_flat"
  fi
  # on KNL save 4 corse for the OS
  SCALING_EXE="-r 4 ${SCALING_EXE}"
  AFFINITY_CHECK_EXE="-r 4 ${AFFINITY_CHECK_EXE}"
elif [ "${ARCH}" == "hsw" ]; then
  HTs=(1 2)
  PEs=(1 2 4 8 16)
  NUM_NUMA=2
  PE_PER_NUMA=16
  THREADS_PER_CORE=2
  FILE_PREFIX="${ROOT_DIR}/runs/hsw"
  SCALING_EXE="${ROOT_DIR}/bin/SolverDriver_hsw_openmp.exe"
  AFFINITY_CHECK_EXE="${ROOT_DIR}/bin/omp_diag.hsw"
  MSUB_NODE_SUFFIX="ppn=32:haswell"
else
 echo "ERROR, unknown: ARCH=${ARCH}"
 exit
fi


mkdir -p ${FILE_PREFIX}

NODAL_NX=246;
NODAL_NY=246;
NODAL_NZ=246;




for num_nodes in 1 2 4 8 16 32 64 90; do
for ht in ${HTs[*]}; do

# pack jobs per node into one job

CUBE_SIZE=`echo "e(l( (${NODAL_NX}*${NODAL_NY}*${NODAL_NZ}*1.0*${num_nodes}) )/3.0)" | bc -l | python -c "from math import ceil; print int(ceil(float(raw_input())))"`
NX=${CUBE_SIZE}
NY=${CUBE_SIZE}
NZ=${CUBE_SIZE}

job_name="${NX}x${NY}x${NZ}-ht${ht}";

FILE=${FILE_PREFIX}/n-${num_nodes}_${job_name}.msub

#determine the walltime. Give 1 hour per PE combo
num_pes=${#PEs[@]}
wall_hours=${num_pes};

WALLTIME="${num_pes}:00:00"

cat > $FILE <<- EOM
#!/bin/bash
#MSUB -l walltime=${WALLTIME}
#MSUB -l nodes=${num_nodes}:${MSUB_NODE_SUFFIX}
#MSUB -N ${job_name}

module rm craype-mic-knl
module rm craype-haswell
module rm intel
module load friendly-testing
module load intel/17.0.1

if [ "${ARCH}" == "knl" ]; then
  module load craype-mic-knl
else
  module load craype-haswell
fi

echo "===========  HOST  ==============="

date
hostname

echo "=========== MODULES ==============="
module list -t
echo "==================================="


######################################################
######################################################
# begin run specific stuff
######################################################
######################################################


EOM


# how many PEs will be allocated to a process
# 64 means a single process per KNL node, which I do not do
for pe in ${PEs[*]}; do

let ppr=PE_PER_NUMA/pe;
let tasks_per_node=ppr*NUM_NUMA;

let num_threads=pe*ht;

job_decomp="${num_nodes}x${tasks_per_node}x${pe}x${ht}"


let TOTAL_MPI_PROCS=num_nodes*tasks_per_node




cat >> $FILE <<- EOM

######################################################
# job: ${job_decomp}

cd ${FILE_PREFIX}

uuid=\$(uuidgen)
mkdir \${uuid};
cd \${uuid}
pwd

touch \${PBS_JOBID}-${job_name}-${job_decomp}

# get the gold file and config
cp -Lr ${GOLD_DIR} .
cp ${DRIVER_XML} .


if [ "${ht}" == "1" ]; then
WAIT_POLICY=active
else
WAIT_POLICY=passive
fi

NODE_ENV=\$(aprun \
-e OMP_NUM_THREADS=${num_threads} \
-e OMP_DISPLAY_ENV=verbose \
-e KMP_AFFINITY=verbose,warnings,respect,granularity=core,duplicates,compact,0,0 \
-e OMP_WAIT_POLICY=\${WAIT_POLICY} \
-d ${num_threads} \
-j ${ht} \
-cc depth \
-N 1 \
-n 1 \
env)

echo "\${NODE_ENV}"  > ${job_name}.env

echo "Gathering affinity information"

aprun \
-e OMP_NUM_THREADS=${num_threads} \
-e OMP_DISPLAY_ENV=verbose \
-e KMP_AFFINITY=verbose,warnings,respect,granularity=core,duplicates,compact,0,0 \
-e OMP_WAIT_POLICY=\${WAIT_POLICY} \
-d ${num_threads} \
-j ${ht} \
-cc depth \
-N ${tasks_per_node} \
-n ${TOTAL_MPI_PROCS} \
${AFFINITY_CHECK_EXE}

mv affinity_details.csv affinity-${job_name}-${job_decomp}.csv

# echo the following
set -x

aprun \
-e OMP_NUM_THREADS=${num_threads} \
-e OMP_DISPLAY_ENV=verbose \
-e KMP_AFFINITY=verbose,warnings,respect,granularity=core,duplicates,compact,0,0 \
-e OMP_WAIT_POLICY=\${WAIT_POLICY} \
-d ${num_threads} \
-j ${ht} \
-cc depth \
-N ${tasks_per_node} \
-n ${TOTAL_MPI_PROCS} \
${SCALING_EXE} \
--xml=driver.xml \
--nx=${NX} --ny=${NY} --nz=${NZ} \
--matrixType=Laplace3D \
--cores_per_proc=${pe} --threads_per_core=${ht}

# stop echoing
set +x

# n total MPI tasks
# N node num MPI tasks
# r reserve cores for OS

date

EOM

# end of PEs
done

echo $FILE

# end of HT
done

# end of nodes
done

# end of HSW/KNL loop
done
