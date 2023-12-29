#!/bin/bash

# Run this script from the project root directory
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

pdm install
pdm add mpi4py

# Add amisc package (check for local editable installation)
if [ -d "../amisc" ]; then
    echo "Adding ../amisc in editable mode..."
    pdm remove amisc
    pdm add -e ../amisc --dev
fi

# Run juliacall update (can't do in sbatch for some GitError)
echo "Initializing juliacall..."
pdm run python -c "import juliacall"

# Add slurm user account info to .bashrc
if [[ -z "${SLURM_ACCOUNT}" || -z "${SLURM_MAIL}" ]]; then
  case $(whoami) in
    eckelsjd)
      export SLURM_ACCOUNT='goroda0'
      export SLURM_MAIL='eckelsjd@umich.edu'
      ;;
    mgallen)
      export SLURM_ACCOUNT='bjorns0'
      export SLURM_MAIL='mgallen@umich.edu'
      ;;
    *)
      export SLURM_ACCOUNT='goroda0'
      export SLURM_MAIL='eckelsjd@umich.edu'
      ;;
  esac

  echo "export SLURM_ACCOUNT=${SLURM_ACCOUNT}" >> ~/.bashrc
  echo "export SLURM_MAIL=${SLURM_MAIL}" >> ~/.bashrc
fi

echo "Environment setup complete! Use 'pdm train <scripts_folder>' to build a surrogate."
