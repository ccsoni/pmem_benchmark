#!/bin/bash
#PBS -A RTRHD
#PBS -q gpu
#PBS -b 2
#PBS -l elapstim_req=00:05:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.5/llvm-intel2022.3.1-cuda12.1.0
#PBS -v OMP_NUM_THREADS=48
#PBS -v USE_DEVDAX=pmemkv
#PBS -v RDMAV_HUGEPAGES_SAFE=1

module load openmpi/$NQSV_MPI_VER

cd ${PBS_O_WORKDIR}

mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 10
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 12
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 14
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 16
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 18
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 20
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 22
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 24
mpirun ${NQSV_MPIOPTS} -np 2 -npernode 1 --bind-to none ./bcast 26
