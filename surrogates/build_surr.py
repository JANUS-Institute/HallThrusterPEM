import sys
import dill
from pathlib import Path
import pickle
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor
import numpy as np

sys.path.append('..')

from models.pem import pem_system


def gen_test_set():
    N = 1100
    sys = pem_system(executor=None, init=False)
    xt = sys.sample_exo_inputs((N,))     # (N, xdim)
    yt = sys(xt, ground_truth=True, training=False)
    nan_idx = np.any(np.isnan(yt), axis=-1)
    xt = xt[~nan_idx, :]
    yt = yt[~nan_idx, :]
    data = {'xt': xt, 'yt': yt}

    with open(Path(sys.root_dir) / 'test_set.pkl', 'wb') as dill_file:
        dill.dump(data, dill_file)


def train():
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            sys = pem_system(executor=executor, init=True)  # Initializes with coarsest fidelity indices
            end_time_s = 1*3600  # 2 days
            with open(Path('../models') / 'thruster_svd.pkl', 'rb') as fd:
                d = pickle.load(fd)
                r1 = d['vtr'].shape[0]
            with open('test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            qoi_ind = [1, 2, 7, 8]  # just I_B0, thrust, and first 2 u_ion svd coeffs
            # qoi_ind = [0, 1, 2, 7, 8, int(7 + r1), int(7 + r1 + 1), int(7 + r1 + 2)]
            sys.build_system(qoi_ind=qoi_ind, N_refine=100, max_iter=20, max_tol=1e-8, max_runtime=end_time_s,
                             save_interval=5, test_set=test_set)


if __name__ == '__main__':
    # gen_test_set()
    train()
