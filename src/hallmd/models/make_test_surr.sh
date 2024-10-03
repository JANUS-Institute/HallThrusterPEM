#!/bin/bash

#SBATCH --job-name=make_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --time=0:30:00
#SBATCH --output=./logs/make_test.%j.out
#SBATCH --error=./logs/make_test.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

pdm run python pem.py
