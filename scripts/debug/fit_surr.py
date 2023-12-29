"""Debugging script to be used with `fit_surr.sh` to test submitting small SLURM jobs."""
import sys
import time

import dill
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor
from concurrent.futures import wait, ALL_COMPLETED
from joblib import Parallel, delayed, cpu_count

from hallmd.models.pem import pem_v0


def print_hello(rank):

    def func(i):
        return i + rank

    res = Parallel(n_jobs=-1, verbose=9)(delayed(func)(i) for i in range(10))
    msg = f'Process {rank} on node {MPI.Get_processor_name()} completed with cpus={cpu_count()}. Ret = {res}\n'
    time.sleep(3)
    sys.stdout.write(msg)

    return rank*2


if __name__ == '__main__':
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    print(f'Starting MPI training...')
    print(f'Size: {size}. Rank: {rank}. Name: {name}.')

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            fs = [executor.submit(print_hello, i) for i in range(5)]
            wait(fs, timeout=None, return_when=ALL_COMPLETED)
            for i, future in enumerate(fs):
                print(f'i={i}: {future.result()}')

            print('Master has finished MPI training!')
