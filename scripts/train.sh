#!/bin/bash
# This script is used to generate data and fit a surrogate model on an HPC cluster (using SLURM); in 3 steps:
#
# 1) gen - Generate compression and test set data (gen_data.py)
# 2) fit - Train the surrogate model (fit_surr.py)
# 3) slice - Plot 1d slices of the surrogate model (plot_slice.py)
#
# Pass --local to run on a local machine instead of an HPC cluster.
# Pass any of --skip-gen, --skip-fit, or --skip-slice to skip the corresponding step.
# Pass --gen-cpus, --fit-cpus, or --slice-cpus to specify the number of CPUs to use for each step (always on 1 node).
# Pass --mem-per-cpu to specify the memory per CPU for each step (same for all)
# All command line arguments are passed to all Python scripts. See their docs for details.
#
# Usage: ./train.sh [PYTHON_ARGS] --local --skip-gen --skip-fit --skip-slice --mem-per-cpu 3g etc.
#
# Author: Joshua Eckels
# Date: 12/3/2024

set -e
args=()

PYTHON_VERSION=3.11.5
GCC_VERSION=10.3.0
OPENMPI_VERSION=4.1.6

if command -v module &> /dev/null; then
    echo "Detected module command. Loading modules..."
    module load python/${PYTHON_VERSION}
    module load gcc/${GCC_VERSION}
    module load openmpi/${OPENMPI_VERSION}
    export MPICC=$(which mpicc)
fi

# Parse the command line arguments (surely there is a better way to do this...)
SKIP_GEN=false
SKIP_FIT=false
SKIP_SLICE=false
GEN_CPUS=36
FIT_CPUS=16
SLICE_CPUS=36
MEM_PER_CPU=3g
LOCAL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-gen) SKIP_GEN=true ;;
        --skip-fit) SKIP_FIT=true ;;
        --skip-slice) SKIP_SLICE=true ;;
        --gen-cpus) GEN_CPUS=$2; shift ;;
        --fit-cpus) FIT_CPUS=$2; shift ;;
        --slice-cpus) SLICE_CPUS=$2; shift ;;
        --mem-per-cpu) MEM_PER_CPU=$2; shift ;;
        --local) LOCAL=true ;;
        --search|-s) : ;;
        *) args+=("$1") ;;
    esac
    shift  # Remove the current argument from the list
done

if command -v sbatch &> /dev/null && [[ "$LOCAL" == "false" ]]; then
    echo "Detected sbatch command. Submitting jobs to the HPC cluster..."

    # Generate compression and test set data
    if [[ "$SKIP_GEN" == "false" ]]; then
        job1_id=$(sbatch --mail-user="${SBATCH_MAIL_USER}" \
                         --mail-type=ALL \
                         --job-name=gen_data \
                         --partition=standard \
                         --time=00-02:00:00 \
                         --nodes=1 \
                         --mem-per-cpu="${MEM_PER_CPU}" \
                         --ntasks-per-node=1 \
                         --cpus-per-task="${GEN_CPUS}" \
                         --wrap="set -e; pdm run python gen_data.py ${args[*]}" | awk '{print $4}')
        echo "Gen data job submitted with Job ID: $job1_id"
    else
        echo "Skipping data generation."
    fi

    # Fit the surrogate when the data generation job is complete and successful
    if [[ "$SKIP_FIT" == "false" ]]; then
        job2_id=$(sbatch --mail-user="${SBATCH_MAIL_USER}" \
                         --mail-type=ALL \
                         --job-name=fit_surr \
                         --partition=standard \
                         --time=00-07:00:00 \
                         --nodes=1 \
                         --mem-per-cpu="${MEM_PER_CPU}" \
                         --ntasks-per-node=1 \
                         --cpus-per-task="${FIT_CPUS}" \
                         --output=./logs/%x-%j.log \
                         --dependency=$([[ "$SKIP_GEN" == "false" ]] && echo "afterok:$job1_id" || echo "singleton") \
                         --wrap="set -e; pdm run python fit_surr.py ${args[*]} --search" | awk '{print $4}')
        echo "Fit surrogate job submitted with Job ID: $job2_id"
        # When more than 36 cpus are needed, use MPI with >1 nodes/tasks and srun:
        # sbatch ... --nodes=2 --ntasks-per-node=36 --cpus-per-task=1
        # srun --cpus-per-task=$SLURM_CPUS_PER_TASK python -m mpi4py.futures fit_surr.py
    else
        echo "Skipping surrogate fit."
    fi

    # Plot 1d slices
    if [[ "$SKIP_SLICE" == "false" ]]; then
        job3_id=$(sbatch --mail-user="${SBATCH_MAIL_USER}" \
                         --mail-type=ALL \
                         --job-name=plot_slice \
                         --partition=standard \
                         --time=00-00:30:00 \
                         --nodes=1 \
                         --mem-per-cpu="${MEM_PER_CPU}" \
                         --ntasks-per-node=1 \
                         --cpus-per-task="${SLICE_CPUS}" \
                         --output=./logs/%x-%j.log \
                         --dependency=$([[ "$SKIP_FIT" == "false" ]] && echo "afterok:$job2_id" || echo "singleton") \
                         --wrap="set -e; pdm run python plot_slice.py ${args[*]} --search" | awk '{print $4}')
        echo "Plot slice job submitted with Job ID: $job3_id"
    else
        echo "Skipping 1d slice plotting."
    fi

    echo "All jobs submitted."

else
    echo "Running on local machine..."

    if [[ "$SKIP_GEN" == "false" ]]; then
        pdm run python gen_data.py "${args[@]}"
        echo "Finished gen_data.py."
    else
        echo "Skipping data generation."
    fi

    if [[ "$SKIP_FIT" == "false" ]]; then
        pdm run python fit_surr.py "${args[@]}" --search
        echo "Finished fit_surr.py."
    else
        echo "Skipping surrogate fit."
    fi

    if [[ "$SKIP_SLICE" == "false" ]]; then
        pdm run python plot_slice.py "${args[@]}" --search
        echo "Finished plot_slice.py."
    else
        echo "Skipping 1d slice plotting."
    fi

    echo "All jobs complete."
fi
