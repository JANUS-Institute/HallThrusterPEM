#!/bin/bash

#SBATCH --job-name=gen_data_debug
#SBATCH --account=goroda0
#SBATCH --partition=debug
#SBATCH --time=00-00:01:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100m
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./scripts/debug/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

# Run this script from the project root directory
echo "Starting job script..."

module load python/3.11.5

# export PYTHON_JULIAPKG_OFFLINE=yes
pdm run python scripts/debug/gen_data.py

echo "Finishing job script..."
