#!/bin/bash
# This script is used to generate data and fit a surrogate model on a local machine.
# It runs the data generation and surrogate fitting in sequence.
# All command line arguments are passed to both scripts. See their docs for details.
#
# Date: 12/3/2024

set -e
pdm run python gen_data.py "$@"
echo "Finished gen_data.py."

pdm run python fit_surr.py "$@" --search
echo "Finished fit_surr.py."
