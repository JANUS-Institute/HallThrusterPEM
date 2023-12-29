#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=fit_surr_debug
#SBATCH --account=goroda0
#SBATCH --partition=debug
#SBATCH --time=00-00:03:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --output=./scripts/debug/logs/%x-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python/3.11.5
module load gcc/10.3.0
module load openmpi/4.1.6
export MPICC=$(which mpicc)
module list

export PYTHON_JULIAPKG_OFFLINE=yes
srun --cpus-per-task=$SLURM_CPUS_PER_TASK pdm run python -m mpi4py.futures scripts/debug/fit_surr.py

echo "Finishing job script..."
