import sys
import dill
from pathlib import Path
import pickle
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor

sys.path.append('..')

from models.pem import pem_system


if __name__ == '__main__':
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            sys = pem_system(executor=executor)  # Initializes with coarsest fidelity indices
            end_time_s = 2*24*3600  # 2 days
            with open(Path('../models') / 'thruster_svd.pkl', 'rb') as fd:
                d = pickle.load(fd)
                r1 = d['vtr'].shape[0]
            qoi_ind = [0, 1, 2, 7, 8, int(7 + r1), int(7 + r1 + 1), int(7 + r1 + 2)]
            sys.build_system(qoi_ind=qoi_ind, N_refine=200, max_iter=1000, max_tol=1e-8, max_runtime=end_time_s,
                             save_interval=50)
