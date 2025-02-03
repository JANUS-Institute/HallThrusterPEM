#!/bin/bash
#SBATCH --job-name=mcmc_newconfig6
#SBATCH --account=goroda98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=output_newconfig6.log
#SBATCH --error=error_newconfig6.log
#SBATCH --mail-type=END,FAIL

pdm run python3 \
	scripts/mcmc.py \
	scripts/pem_v1/pem_v1_SPT-100.yml \
	--max_samples=50000 \
	--datasets macdonald2019 diamant2014 \
	--output_interval=100 \
	--output_dir=/scratch/goroda_root/goroda98/marksta/pem_spt100 \
	--init_sample=scripts/pem_v1/map.csv \
	--init_cov=scripts/pem_v1/cov.csv \
