#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=test_svd
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-03:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.10-anaconda
source $ANACONDA_ROOT/etc/profile.d/conda.sh
conda activate hall_pem

python test_mc.py

echo "Finishing job script..."