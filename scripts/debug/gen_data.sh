#!/bin/bash

#SBATCH --job-name=gen_data_debug
#SBATCH --partition=standard
#SBATCH --time=00-00:03:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./scripts/debug/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL

# Run this script from the project root directory
set -e
echo "Starting job script..."

module load python/3.11.5

export PYTHON_JULIAPKG_OFFLINE=yes
pdm run python scripts/debug/gen_data.py

echo "Finishing job script..."
