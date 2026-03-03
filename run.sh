#!/bin/bash

N=5

DATA="../pem_data/data/spt100"
DIAMANT1="${DATA}/diamant2014/data_aerospace.csv"
DIAMANT2="${DATA}/diamant2014/data_L3.csv"
TENENBAUM="${DATA}/tenenbaum2019/data.csv"

TIMESTAMP="2026-02-25T04.09.12"
MCMC_DIR="C:\Users\marksta\Documents\Projects\hall_thruster_pem\HallThrusterPEM\outputs\pem_${TIMESTAMP}\mcmc"
SAMPLE_FILE="${MCMC_DIR}\samples.csv"

uv run scripts/run_mcmc.py\
    scripts/pem_v1/pem_v1_SPT-100.yml\
    --max-samples=${N}\
    --datasets ${TENENBAUM} \
    --output-interval=${N} \
    --output-dir=outputs \
    --ncharge=1 \
    --noise-std=0.025 \
    --init-sample=${SAMPLE_FILE} \
    --init-cov="${MCMC_DIR}\cov_chol.csv" \
    --sampler="dram" \
