#!/bin/bash
#PBS -A RTRHD
#PBS -q gpu
#PBS -b 4
#PBS -l elapstim_req=00:05:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.5/llvm-intel2022.3.1-cuda12.1.0
#PBS -v OMP_NUM_THREADS=48
#PBS -v USE_DEVDAX=pmemkv
#PBS -v RDMAV_HUGEPAGES_SAFE=1

module load openmpi/$NQSV_MPI_VER

cd ${PBS_O_WORKDIR}

mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 10
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 12
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 14
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 16
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 18
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 20
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 22
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 24
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none ./allgather 26
