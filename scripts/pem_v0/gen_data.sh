#!/bin/bash

#SBATCH --job-name=gen_data_v0
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --output=./scripts/pem_v0/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

# Run this script from the project root directory
echo "Starting job script..."

module load python/3.11.5

# export PYTHON_JULIAPKG_OFFLINE=yes
pdm run python scripts/pem_v0/gen_data.py

echo "Finishing job script..."
