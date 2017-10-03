#!/bin/bash

# set the hugepage size
export hp=2
export hp_name="craype-hugepages${hp}M"

# point to the binary
export RUN_EXE=/global/cscratch1/sd/jjellio/binaries/sep-21-mj3-tpetra/SolverDriver.exe

# define the output directory
export EXEC_DIR="./${hp_name}"


# choose the number of nodes

# this is the correct number.. for testing use something else:
#for NODES in 1 2 4 8 16 32 64 128 256 512 544; do
for NODES in 1 4; do

# determine our cores per proc
# we want 64x1, 32x2, 16x4, 8x8 and 4x16
# which gives the list 1,2,4,8,16
for cores_per_proc in 1 2 4 8 16; do

procs_per_node=$((64/cores_per_proc))


rm -f tmp.pbs
cat >tmp.pbs <<END
#!/bin/bash
#PBS -l select=${NODES}:ncpus=64:mpiprocs=${procs_per_node}:nmics=1
#PBS -l walltime=8:00:00
#PBS -q knl

#PBS -A Project_ID
# ? what is this? PBS -l application=Application_Name

#PBS -N muelu_tpetra

export hp=${hp}
export hp_name=${hp_name}

export RUN_EXE=${RUN_EXE}
export EXEC_DIR=${EXEC_DIR}

export nodes=${NODES}

# determine our cores per proc
export nc=${cores_per_proc}
export nr=${procs_per_node}


source /global/homes/j/jjellio/src/TrilinosDrivers/SolverDriver/JobTemplates/pbs/fused-submission.pbs.core

END

# Submit the tmp job
#chmod +x tmp.pbs
#./tmp.pbs
qsub tmp.pbs

# end of cores_per_proc
done
# end of NODES
done


