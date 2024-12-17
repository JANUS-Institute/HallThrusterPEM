#!/bin/bash
# This script will load necessary modules, install PDM, setup a local venv, and
# install the HallThruster.jl Julia package. This script will also export SBATCH_ACCOUNT and SBATCH_MAIL_USER
# to the user's .bashrc file and the environment based on the user's username.
#
# Usage: source setup_env.sh [INSTALL_HALLTHRUSTER_ARGS]
#
# Required HPC modules (if using module command):
# - python/3.11.5
# - gcc/10.3.0
# - openmpi/4.1.6
#
# Please adjust according to your needs.
#
# Author: Joshua Eckels
# Date: 11/19/2024

PYTHON_VERSION=3.11.5
GCC_VERSION=10.3.0
OPENMPI_VERSION=4.1.6

echo "Setting up environment..."

if command -v module &> /dev/null
then
    echo "Detected module command. Loading modules..."
    module load python/${PYTHON_VERSION}
    module load gcc/${GCC_VERSION}
    module load openmpi/${OPENMPI_VERSION}
    export MPICC=$(which mpicc)
else
    echo "No module command detected. Please load the necessary modules manually."
fi

# Make sure no conda environment is activated
if command -v conda deactivate &> /dev/null
then
    conda deactivate &> /dev/null
fi

# Install PDM and setup local venv
if ! command -v pdm &> /dev/null
then
    echo "Installing PDM..."
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
else
    echo "PDM already installed!"
fi

pdm self update

if command -v module &> /dev/null
then
    pdm sync --prod -G mpi -G scripts -L ../pdm.lock
else
    pdm sync --prod -G scripts -L ../pdm.lock
fi

pdm run python install_hallthruster.py "$@"

# Add slurm user account info to .bashrc
if command -v sbatch &> /dev/null
then
    echo "Detected sbatch command. Setting up SLURM account and email..."
    if [[ -z "${SBATCH_ACCOUNT}" || -z "${SBATCH_MAIL_USER}" ]]; then
      case $(whoami) in
        eckelsjd)
          export SBATCH_ACCOUNT='goroda98'
          export SBATCH_MAIL_USER='eckelsjd@umich.edu'
          ;;
      esac

      echo "export SBATCH_ACCOUNT=${SBATCH_ACCOUNT}" >> ~/.bashrc
      echo "export SBATCH_MAIL_USER=${SBATCH_MAIL_USER}" >> ~/.bashrc
    fi
fi

echo "Environment setup complete! Use 'train.sh' to train a surrogate."
