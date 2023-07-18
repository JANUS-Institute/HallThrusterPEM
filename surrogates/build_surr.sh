#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=build_surr
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-16:00:00
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=4g
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.10-anaconda
module load gcc/10.3.0
module load openmpi/4.1.4
export MPICC=$(which mpicc)
module list

source $ANACONDA_ROOT/etc/profile.d/conda.sh
conda activate hall_pem
conda info --envs

export PYTHON_JULIAPKG_OFFLINE=yes
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python -m mpi4py.futures build_surr.py

echo "Finishing job script..."
