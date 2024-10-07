#!/bin/bash

#SBATCH --job-name=mcmc_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --time=10:00:00
#SBATCH --output=./logs/mcmc_job.%j.out
#SBATCH --error=./logs/mcmc_job.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# --job-name=mcmc_job: Sets the job name.
# --nodes=1: Specifies that the job will use 1 node.
# --ntasks-per-node=1: Specifies the number of tasks per node (usually 1 for a single job).
# --cpus-per-task=32: Specifies the number of CPU cores per task. Adjust based on your HPC resources.
# --mem=64G: Specifies the memory required per node. Adjust based on your needs.
# --partition=standard: Specifies the partition to submit the job to. Adjust based on your HPC setup.
# --time=48:00:00: Sets the time limit for the job.
# --output=mcmc_job.%j.out: Specifies the standard output file.
# --error=mcmc_job.%j.err: Specifies the standard error file

# MLE: 1 Hour, 1G mem
# MCMC: debugging = 2 Hour, 5GB

# Run the MCMC script
pdm run python mcmc.py
