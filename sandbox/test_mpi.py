from mpi4py import MPI
from concurrent.futures import wait, ALL_COMPLETED
from mpi4py.futures import MPICommExecutor
import sys
from joblib import Parallel, delayed, cpu_count
import time


def print_hello(rank):

    def func(i):
        return i + rank

    res = Parallel(n_jobs=-1, verbose=9)(delayed(func)(i) for i in range(100))
    msg = f'Process {rank} on node {MPI.Get_processor_name()} completed with cpus={cpu_count()}. Ret = {res}\n'
    time.sleep(3)
    sys.stdout.write(msg)

    return rank*2


if __name__ == '__main__':
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            fs = [executor.submit(print_hello, i) for i in range(10)] 
            wait(fs, timeout=None, return_when=ALL_COMPLETED)
            for i, future in enumerate(fs):
                print(f'i={i}: {future.result()}')

            print('Master has finished MPI execution')

