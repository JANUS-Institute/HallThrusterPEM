#!/bin/bash
# This script is only setup to run on the University of Michigan Great Lakes cluster.
# The cluster has OpenMPI installed along with the Module package and the SLURM workload manager.
# This script will export SBATCH_ACCOUNT and SBATCH_MAIL_USER to the user's .bashrc file and the environment
# based on the user's username.
#
# Run as:
#
# source setup_hpc.sh --julia-version 1.10 --hallthruster-version 0.17.2
#
# Please adjust according to your needs.
#
# Author: Joshua Eckels
# Date: 11/19/2024

echo "Setting up environment..."

module load python/3.11.5
module load gcc/10.3.0
module load openmpi/4.1.6
export MPICC=$(which mpicc)
module list

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
pdm sync --prod -G mpi -G scripts

pdm run python install_hallthruster.py "$@"

# Add slurm user account info to .bashrc
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

echo "Environment setup complete! Use 'pdm train <config_file>' to build a surrogate."
