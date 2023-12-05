#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=build_madison_surrogate2
#SBATCH --account=bjorns0
#SBATCH --partition=standard
#SBATCH --time=00-05:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2g
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output="./multi3_output%x-%j.log"
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mgallen@umich.edu

echo "Starting job script..."



export PYTHON_JULIAPKG_OFFLINE=yes
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python -m mpi4py.futures madison.py

echo "Finishing job script..."
