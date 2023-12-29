#!/bin/bash

# Run this script from the project root directory
echo "Setting up environment..."

module load python/3.11.5
module load gcc/10.3.0
module load openmpi/4.1.6
export MPICC=$(which mpicc)
module list

# Make sure no conda environment is activated
if command -v conda &> /dev/null
then
    conda deactivate
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

# Add amisc package (check for local editable installation)
if [ -d "../amisc" ]; then
    echo "Adding ../amisc in editable mode..."
    pdm remove amisc
    pdm add -e ../amisc --dev
fi

# Run juliacall update (can't do in sbatch for some GitError)
pdm run python -c "import juliacall"

echo "Environment setup complete! Use 'pdm train <scripts_folder>' to build a surrogate."
