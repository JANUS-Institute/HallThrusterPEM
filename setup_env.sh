#!/bin/bash

# Run this script from the project root directory
echo "Setting up environment..."

module load python/3.11.5

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

echo "Environment setup complete! Use 'pdm train <scripts_folder>' to build a surrogate."