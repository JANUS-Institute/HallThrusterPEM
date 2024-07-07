#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=fit_surr_v0
#SBATCH --partition=standard
#SBATCH --time=00-12:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5g
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --export=ALL
#SBATCH --output=./scripts/pem_v0/logs/%x-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
echo "Starting job script..."

module load python/3.11.5
module load gcc/10.3.0
module load openmpi/4.1.6
export MPICC=$(which mpicc)
module list

export PYTHON_JULIAPKG_OFFLINE=yes
srun --cpus-per-task=$SLURM_CPUS_PER_TASK pdm run python -m mpi4py.futures scripts/pem_v0/fit_surr.py

echo "Finishing job script..."
