#!/bin/bash
#SBATCH -p regular
#SBATCH --reboot
#SBATCH -C knl,quad,cache
#SBATCH --exclusive
#SBATCH --zonesort=180
#SBATCH -J rebooted-muelu
#SBATCH -t 02:00:00
#SBATCH --core-spec=4
# --array=1,2,4,8,16,32,64
#SBATCH --array=1,2,4,8,16
#SBATCH -vvv

date

RUN_EXE=/global/cscratch1/sd/jjellio/binaries/sep-21-mj3-tpetra/SolverDriver.exe

SRUN_REQ_TIME=1;
start_ts=`SLURM_TIME_FORMAT="%s"  squeue -j ${SLURM_JOBID}  -o %S --noheader`;
end_ts=`SLURM_TIME_FORMAT="%s"  squeue -j ${SLURM_JOBID}  -o %e --noheader`;
let required_seconds_per_srun=${SRUN_REQ_TIME}*60;

nodes=${SLURM_JOB_NUM_NODES}
execSpace=openmp

# determine our cores per proc
nc=${SLURM_ARRAY_TASK_ID}
nr=$((64/nc))

##### Setup Hugepages
# use 2M hugepages
export hp=2

module -s rm craype-hugepages512M craype-hugepages256M craype-hugepages128M craype-hugepages64M craype-hugepages32M craype-hugepages16M craype-hugepages8M craype-hugepages2M craype-hugepages4M
hp_name="craype-hugepages${hp}M"

module load ${hp_name}

export MPICH_ALLOC_MEM_HUGE_PAGES=1
export MPICH_ALLOC_MEM_PG_SZ=${hp}M
export MPICH_ENV_DISPLAY=verbose
export HUGETLB_VERBOSE=2


exec_dir="${SCRATCH}/muelu-cache-reboot-compact-switches/${hp_name}"
echo "exec_dir: ${exec_dir}"
mkdir -p ${exec_dir} 2>&1 > /dev/null

ln -s $(realpath gold) ${exec_dir} 2>&1 > /dev/null
ln -s $(realpath fused-driver.xml) ${exec_dir} 2>&1 > /dev/null

pushd ${exec_dir}

# track node uptime
srun --ntasks-per-node=1 bash -c "uptime > \`hostname\`.${SLURM_JOB_ID}.uptime"


for PROBLEM_TYPE in "Laplace3D"; do
#PROBLEM_TYPE="Brick3D"
NODAL_NX=246;
NODAL_NY=246;
NODAL_NZ=246;

if [ "${PROBLEM_TYPE}" == "Elasticity3D" ]; then

NODAL_NX=171;
NODAL_NY=171;
NODAL_NZ=171;

fi


CUBE_SIZE=`echo "e(l( (${NODAL_NX}*${NODAL_NY}*${NODAL_NZ}*1.0*${nodes}) )/3.0)" | bc -l | python -c "from math import ceil; print int(ceil(float(raw_input())))"`

# 64 procs per node also does flat MPI run
if [ "${nr}" == "64" ]; then

execSpace=serial
current_ts=`date +"%s"`;

# how many seconds remain
let remaining_ts=${end_ts}-${current_ts};

if [ "${remaining_ts}" -lt "${required_seconds_per_srun}" ]; then
  # not enough time to run
  echo "Not enough time for another srun"
  exit;
fi

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#export OMP_DISPLAY_ENV=verbose
#export KMP_AFFINITY=verbose

export procs_per_node=${nr}
export cores_per_proc=${nc}
export threads_per_core=1


echo "${hp_name} : $nodes, $procs_per_node, $cores_per_proc $threads_per_core : ${CUBE_SIZE}x${CUBE_SIZE}x${CUBE_SIZE}"

read -r -d '' OMP_STUFF <<- EOM
OpenMP:
  OMP_NUM_THREADS=${OMP_NUM_THREADS}
  OMP_PLACES=${OMP_PLACES}
  OMP_PROC_BIND=${OMP_PROC_BIND}
EOM

echo "$OMP_STUFF"

let ntasks=nodes*nr

fname=weakscaling_${execSpace}_${PROBLEM_TYPE}-${CUBE_SIZE}x${CUBE_SIZE}x${CUBE_SIZE}_decomp-${nodes}x${procs_per_node}x${cores_per_proc}x${threads_per_core}_knl-quad-cache-rebooted.log

srun_cmd="--nodes=${nodes} \
--ntasks-per-node=${procs_per_node} \
-c $(( 256 / ${procs_per_node} )) --cpu_bind=cores \
numactl -m 0 \
${RUN_EXE} --xml=fused-driver.xml \
--node=${execSpace} \
--matrixType=${PROBLEM_TYPE} \
--nx=${CUBE_SIZE} --ny=${CUBE_SIZE} --nz=${CUBE_SIZE} \
--Threads_per_core=${threads_per_core} \
--cores_per_proc=${cores_per_proc}"

echo "srun ${srun_cmd}"

if [ -s "${fname}-ok" ]; then
  echo "Skipping nr=${nr} nht=${nht}, file, $fname exists"
else
srun --ntasks-per-node=1 bash -c "env > ${fname}.${SLURM_JOB_ID}.`hostname`.env"

time srun ${srun_cmd} |& tee ${fname}

mv ${fname} ${fname}-ok

fi

# end of flat MPI run
fi


tar czf ${SLURM_JOB_ID}.uptime.tar.gz *.${SLURM_JOB_ID}.uptime
rm -f *.${SLURM_JOB_ID}.uptime


# loop over HT combos
for nht in 1 2 4; do

execSpace=openmp
current_ts=`date +"%s"`;

# how many seconds remain
let remaining_ts=${end_ts}-${current_ts};

if [ "${remaining_ts}" -lt "${required_seconds_per_srun}" ]; then
  # not enough time to run
  echo "Not enough time for another srun"
  exit;
fi


export OMP_NUM_THREADS=$(( ${nc}*${nht} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#export OMP_DISPLAY_ENV=verbose
#export KMP_AFFINITY=verbose

export procs_per_node=${nr}
export cores_per_proc=${nc}
export threads_per_core=${nht}


echo "${hp_name} : $nodes, $procs_per_node, $cores_per_proc $threads_per_core : ${CUBE_SIZE}x${CUBE_SIZE}x${CUBE_SIZE}"


read -r -d '' OMP_STUFF <<- EOM
OpenMP:
  OMP_NUM_THREADS=${OMP_NUM_THREADS}
  OMP_PLACES=${OMP_PLACES}
  OMP_PROC_BIND=${OMP_PROC_BIND}
EOM

echo "$OMP_STUFF"

let ntasks=nodes*nr

fname=weakscaling_${execSpace}_${PROBLEM_TYPE}-${CUBE_SIZE}x${CUBE_SIZE}x${CUBE_SIZE}_decomp-${nodes}x${procs_per_node}x${cores_per_proc}x${threads_per_core}_knl-quad-cache-rebooted.log

srun_cmd="--nodes=${nodes} \
--ntasks-per-node=${procs_per_node} \
-c $(( 256 / ${procs_per_node} )) --cpu_bind=cores \
numactl -m 0 \
${RUN_EXE} --xml=fused-driver.xml \
--node=${execSpace} \
--matrixType=${PROBLEM_TYPE} \
--nx=${CUBE_SIZE} --ny=${CUBE_SIZE} --nz=${CUBE_SIZE} \
--Threads_per_core=${threads_per_core} \
--cores_per_proc=${cores_per_proc}"

echo "srun ${srun_cmd}"

if [ -s "${fname}-ok" ]; then
  echo "Skipping nr=${nr} nht=${nht}, file, $fname exists"
else
srun --ntasks-per-node=1 bash -c "env > ${fname}.${SLURM_JOB_ID}.`hostname`.env"

time srun ${srun_cmd} |& tee ${fname}

mv ${fname} ${fname}-ok

fi

# HT runs
done

# problem type
done



popd

date


