#!/bin/bash
# Script for debugging with sbatch on HPC.
#
# Call as:
#
# sbatch debug.sh <sbatch-args> <debug.py-args>

#SBATCH --job-name=debug
#SBATCH --partition=debug
#SBATCH --time=00-00:02:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/%x-%j.log

set -e
echo "Starting job script..."

module load python/3.11.5

pdm run python debug.py "$@"

echo "Finishing job script..."
