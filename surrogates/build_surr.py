import sys

sys.path.append('..')

from models.pem import pem_system


def build_pem_v0():
    sys = pem_system()  # Initializes with coarsest fidelity indices
    end_time_s = 2*24*3600  # 2 days
    sys.build_system(qoi_ind=None, N_refine=100, max_iter=5, max_tol=0.001, max_runtime=end_time_s, save_interval=1)


if __name__ == '__main__':
    build_pem_v0()
