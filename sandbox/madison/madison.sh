#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=build_madison_surrogate1
#SBATCH --account=bjorns0
#SBATCH --partition=standard
#SBATCH --time=00-10:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2g
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output="./output.log" #/scratch/bjorns_root/bjorns0/mgallen/HallThrusterPEM/sandbox/madison/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mgallen@umich.edu

echo "Starting job script..."

module load python3.10-anaconda
module load gcc/10.3.0
module load openmpi/4.1.6
export MPICC=$(which mpicc)

source $ANACONDA_ROOT/etc/profile.d/conda.sh
conda activate hall_pem

export PYTHON_JULIAPKG_OFFLINE=yes
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python -m mpi4py.futures madison.py

echo "Finishing job script..."
