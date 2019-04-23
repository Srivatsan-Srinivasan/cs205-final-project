#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p doshi-velez
#SBATCH -n 16
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000
source activate CS205
srun -n $SLURM_NTASKS --mpi=pmi2 python mpi4py_test.py
