#!/bin/bash

#SBATCH --job-name=full_mc_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --time=1:00:00
#SBATCH --output=./logs/mc_job.%j.out
#SBATCH --error=./logs/mc_job.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Run the MC script
pdm run python monte_carlo.py
