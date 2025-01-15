#!/bin/bash
./train.sh pem_test/pem_test.yml -c100 -t100 -e thread -r1 -i100 -f both -N25 -m 1e-6 \
                                 --targets T I_B0 I_d u_ion \
                                 --inputs P_b V_a mdot_a a_1 \
                                 --outputs T I_B0 I_d u_ion \
                                 --show-model best worst \
                                 -w 12 "$@"