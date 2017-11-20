#!/bin/bash

# set the hugepage size
export hp=2
export hp_name="craype-hugepages${hp}M"

# point to the binary
#export RUN_EXE=/home/jjellio/tests/SolverDriver.exe
#cp -a ${RUN_EXE} /gpfs/mira-home/jjellio
export RUN_EXE=/gpfs/mira-home/jjellio/SolverDriver.exe

# define the output directory
export EXEC_DIR="./${hp_name}"


# choose the number of nodes

# this is the correct number.. for testing use something else:
#for NODES in 1 2 4 8 16 32 64 128 256 512 544; do
for NODES in 1; do

# determine our cores per proc
# we want 64x1, 32x2, 16x4, 8x8 and 4x16
# which gives the list 1,2,4,8,16
#for cores_per_proc in 1 2 4 8 16; do
for cores_per_proc in 64; do
procs_per_node=$((64/cores_per_proc))


rm -f tmp.pbs
cat >tmp.pbs <<END
#!/bin/bash
#COBALT -q debug-cache-quad
#COBALT --time 00:30:00
#COBALT -n ${NODES}
#cobalt --attrs mcdram=cache:numa=quad
#COBALT -A MueLu

export hp=${hp}
export hp_name=${hp_name}

export RUN_EXE=${RUN_EXE}
export EXEC_DIR=${EXEC_DIR}

export nodes=${NODES}

# determine our cores per proc
export nc=${cores_per_proc}
export nr=${procs_per_node}

source /home/jjellio/src/TrilinosDrivers/SolverDriver/JobTemplates/cobalt/fused-submission.pbs.core

END

echo "${NODES}x${procs_per_node}x${cores_per_proc}"

# Submit the tmp job
chmod +x tmp.pbs
#./tmp.pbs
qsub --mode script ./tmp.pbs

# end of cores_per_proc
done
# end of NODES
done


