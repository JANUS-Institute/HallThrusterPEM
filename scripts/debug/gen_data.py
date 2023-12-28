"""Debugging script to be used with `gen_data.sh` to test submitting small SLURM jobs."""
import time

from hallmd.models.pem import pem_v0


if __name__ == "__main__":
    print('Generating data...')
    time.sleep(10)
    print('Data generation complete!')
