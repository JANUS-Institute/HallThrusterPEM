#!/bin/bash
# This script is used to generate data and fit a surrogate model on a remote HPC cluster.
# It submits two jobs: one to generate test set data and one to fit the surrogate model.
# The second job depends on the first job and will only run if the first job is successful.
# All command line arguments are passed to both scripts. See their docs for details.
#
# This script is only set up to run on the Great Lakes cluster at the University of Michigan.
#
# Please adjust according to your needs.
#
# Author: Joshua Eckels
# Date: 11/19/2024

module load python/3.11.5

# Capture all extra arguments
EXTRA_ARGS="$*"

# Generate compression and test set data
job1_id=$(pdm run_job --job-name=gen_data \
                      --partition=standard \
                      --time=00-04:00:00 \
                      --nodes=1 \
                      --mem-per-cpu=2g \
                      --ntasks-per-node=1 \
                      --cpus-per-task=36 \
                      --output=./logs/%x-%j.log \
                      --wrap="set -e; python gen_data.py $EXTRA_ARGS" | awk '{print $4}')

echo "Gen data job submitted with Job ID: $job1_id"

# Fit the surrogate when the data generation job is complete and successful
job2_id=$(pdm run_job --job-name=fit_surr \
                      --partition=standard \
                      --time=00-10:00:00 \
                      --nodes=1 \
                      --mem-per-cpu=4g \
                      --ntasks-per-node=1 \
                      --cpus-per-task=36 \
                      --output=./logs/%x-%j.log \
                      --dependency=afterok:$job1_id \
                      --wrap="set -e; python fit_surr.py $EXTRA_ARGS --search" | awk '{print $4}')

# When more than 36 cpus are needed, use MPI with >1 nodes/tasks and srun:
#
# sbatch ... --nodes=2 --ntasks-per-node=36 --cpus-per-task=1
#
# module load gcc/10.3.0
# module load openmpi/4.1.6
# export MPICC=$(which mpicc)
#
# srun --cpus-per-task=$SLURM_CPUS_PER_TASK python -m mpi4py.futures fit_surr.py

echo "Fit surrogate job submitted with Job ID: $job2_id"
