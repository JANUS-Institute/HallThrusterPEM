#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=test_mpi
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-00:01:00
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=100m
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/HallThrusterPEM/sandbox/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.10-anaconda
module load gcc/10.3.0
module load openmpi/4.1.4
export MPICC=$(which mpicc)

source $ANACONDA_ROOT/etc/profile.d/conda.sh
conda activate hall_pem

srun --cpus-per-task=$SLURM_CPUS_PER_TASK python test_mpi.py

echo "Finishing job script..."
