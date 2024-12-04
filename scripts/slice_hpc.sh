#!/bin/bash
# This script is used to compare the surrogate to the true model via 1d slices over the inputs.
# All command line arguments are passed to the plot_slice.py script. See its docs for details.
# This script is only set up to run on the Great Lakes cluster at the University of Michigan.
#
# Please adjust according to your needs.
#
# Author: Joshua Eckels
# Date: 11/19/2024

module load python/3.11.5

# Capture all extra arguments
EXTRA_ARGS="$*"

# Run plot slices python script
job1_id=$(pdm run_job --job-name=plot_slice \
                      --partition=standard \
                      --time=00-01:00:00 \
                      --nodes=1 \
                      --mem-per-cpu=1g \
                      --ntasks-per-node=1 \
                      --cpus-per-task=36 \
                      --output=./logs/%x-%j.log \
                      --wrap="set -e; python plot_slice.py $EXTRA_ARGS" | awk '{print $4}')

echo "Plot slice job submitted with Job ID: $job1_id"
