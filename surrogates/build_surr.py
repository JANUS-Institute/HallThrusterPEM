import sys
import dill
from pathlib import Path
import pickle
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor
import numpy as np
import datetime
from datetime import timezone
import os
import matplotlib.pyplot as plt

sys.path.append('..')

from models.pem import pem_system
from utils import ax_default


def gen_test_set():
    N = 1100
    sys = pem_system(executor=None, init=False)
    xt = sys.sample_inputs((N,))     # (N, xdim)
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
            with open(Path('../models/data') / 'test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            qoi_ind = [1, 2, 8]  # just I_B0, thrust, and first u_ion svd coeff
            # qoi_ind = [0, 1, 2, 7, 8, int(7 + r1), int(7 + r1 + 1), int(7 + r1 + 2)]

            # Set up multi-fidelity vs. single-fidelity comparison folders
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            root_dir = Path('../results/surrogates') / f'mf_{timestamp}'
            os.mkdir(root_dir)
            os.mkdir(root_dir / 'single-fidelity')
            os.mkdir(root_dir / 'multi-fidelity')

            # Single-fidelity build
            sf_sys = pem_system(executor=executor, init=True, root_dir=root_dir / 'single-fidelity', hf_override=True)
            sf_sys.build_system(qoi_ind=qoi_ind, N_refine=5000, max_iter=200, max_tol=1e-4, max_runtime=3,
                                save_interval=50, test_set=test_set, prune_tol=1e-8, n_jobs=-1)

            # Multi-fidelity build
            mf_sys = pem_system(executor=executor, init=True, root_dir=root_dir / 'multi-fidelity', hf_override=False)
            mf_sys.build_system(qoi_ind=qoi_ind, N_refine=5000, max_iter=200, max_tol=1e-4, max_runtime=3,
                                save_interval=50, test_set=test_set, prune_tol=1e-8, n_jobs=-1)

            # Get cost allocation for sf and mf systems
            sf_test = sf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            sf_alloc, sf_cum = sf_sys.get_allocation()
            hf_alloc = sf_alloc['Thruster'][str(tuple())]   # [Neval, total cost]
            hf_model_cost = hf_alloc[1] / hf_alloc[0]
            mf_test = mf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            mf_alloc, mf_cum = mf_sys.get_allocation()
            base_alpha = (0,) * len(mf_sys['Thruster'].truth_alpha)
            base_beta = (0,) * len(mf_sys['Thruster'].x_vars)
            base_cost = mf_sys['Thruster'].get_cost(base_alpha, base_beta)

            # Append initial startup costs
            sf_cum = np.insert(sf_cum, 0, hf_model_cost)    # 1 hf evaluation at initialization
            mf_cum = np.insert(mf_cum, 0, base_cost)        # 1 lowest-fidelity evaluation

            # Plot QoI L2 error on test set vs. cost
            labels = [sf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
            fig, axs = plt.subplots(1, len(qoi_ind))
            for i in range(len(qoi_ind)):
                ax = axs[i] if len(qoi_ind) > 1 else axs
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='MF')
                ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='SF')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.grid()
                ax.set_title(labels[i])
                ax_default(ax, r'Cost', r'Relative $L_2$ error', legend=True)
            fig.set_size_inches(3.5*len(qoi_ind), 3.5)
            fig.tight_layout()
            fig.savefig(root_dir/'error_v_cost.png', dpi=300, format='png')

            # Bar chart showing cost allocation breakdown for MF system at end
            fig, ax = plt.subplots()
            width = 0.7
            x = np.arange(len(mf_alloc))
            xlabels = list(mf_alloc.keys())
            cmap = plt.get_cmap('viridis')
            for j, (node, alpha_dict) in enumerate(mf_alloc.items()):
                bottom = 0
                c_intervals = np.linspace(0, 1, len(alpha_dict))
                for i, (alpha, cost) in enumerate(alpha_dict.items()):
                    p = ax.bar(x[j], cost[1] / mf_cum[-1], width, color=cmap(c_intervals[i]), linewidth=1,
                               edgecolor=[0, 0, 0], bottom=bottom)
                    bottom += cost[1] / mf_cum[-1]
                    ax.bar_label(p, labels=[f'{alpha} - {round(cost[0])}'], label_type='center')
            ax_default(ax, "Components", "Fraction of total cost", legend=False)
            ax.set_xticks(x, xlabels)
            ax.set_xlim(left=-1, right=1)
            fig.tight_layout()
            fig.savefig(root_dir/'mf_allocation.png', dpi=300, format='png')
            plt.show()


if __name__ == '__main__':
    # gen_test_set()
    train()
