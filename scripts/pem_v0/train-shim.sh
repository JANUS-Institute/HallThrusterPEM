#!/bin/bash
# Shortcut script to call train.sh with most common arguments.
# Run from the /scripts working directory.
#
# Optional:
# -d : discard outliers
# -p : use variable PDF weighting
./train.sh pem_v0/pem_v0_SPT-100.yml -c500 -t500 -e thread -r3 -i200 -f both -N25 \
                                     --targets T I_B0 I_d u_ion \
                                     --inputs P_b V_a a_1 a_2 \
                                     --outputs T I_B0 I_d u_ion \
                                     --show-model best worst \
                                     --gen-cpus 36 --fit-cpus 16 --slice-cpus 36 --mem-per-cpu 3g "$@"