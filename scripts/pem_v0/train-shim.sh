#!/bin/bash
# Shortcut script to call train.sh with most common arguments.
# Run from the /scripts working directory.
#
# Optional:
# -d : discard outliers
# -p : use variable PDF weighting
./train.sh pem_v0/pem_v0_SPT-100.yml -c200 -t200 -e process -r1 -i150 -f both -N25 -m 1e-4 -C 5 -n 20 \
                                     --targets T I_B0 I_d u_ion \
                                     --inputs P_b V_a a_1 a_2 \
                                     --outputs T I_B0 I_d u_ion \
                                     --show-model best worst \
                                     --gen-cpus 36 --fit-cpus 16 --slice-cpus 36 \
                                     --gen-time 00-01:00:00 --fit-time 00-02:15:00 --slice-time 00-00:15:00 "$@"