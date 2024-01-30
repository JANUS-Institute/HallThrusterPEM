""" `fit_surr.py`

Script to be used with `fit_surr.sh` for building an MD surrogate with MPI parallelism managed by Slurm on an
HPC system. Runs the training procedure for the PEM v0 surrogate.

!!! Note
    The SVD data for PEM v0 should be generated **first** by running `gen_data.py`.

Includes
--------
- `train_mf()` - build and train a multifidelity surrogate for PEM v0 and compare to a single fidelity surrogate.
"""
import datetime
from datetime import timezone
import os
from pathlib import Path
import pickle

import dill
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor
import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default

from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def train_mf(max_runtime_hr=3):
    """Train and compare MF v. SF surrogates."""
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Set up multi-fidelity vs. single-fidelity surrogates
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            root_dir = Path('results') / f'mf_{timestamp}'
            os.mkdir(root_dir)
            os.mkdir(root_dir / 'single-fidelity')
            os.mkdir(root_dir / 'multi-fidelity')
            sf_sys = pem_v0(executor=executor, init=False, save_dir=root_dir / 'single-fidelity', hf_override=True)
            mf_sys = pem_v0(executor=executor, init=False, save_dir=root_dir / 'multi-fidelity', hf_override=False)

            # Filter test set
            with open(CONFIG_DIR / 'test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            lb = np.array([var.bounds()[0] for var in sf_sys.exo_vars])
            ub = np.array([var.bounds()[1] for var in sf_sys.exo_vars])
            keep_idx = np.all((test_set['xt'] < ub) & (test_set['xt'] > lb), axis=1)
            test_set['xt'] = test_set['xt'][keep_idx, :]
            test_set['yt'] = test_set['yt'][keep_idx, :]

            sf_sys.init_system()
            mf_sys.init_system()

            qoi_ind = ['I_B0', 'T', 'uion0', 'uion1']
            sf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=200, max_runtime=max_runtime_hr,
                       save_interval=50, test_set=test_set, n_jobs=-1)
            mf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=200, max_runtime=max_runtime_hr,
                       save_interval=50, test_set=test_set, n_jobs=-1)

            # Get cost allocation for sf and mf systems
            sf_test = sf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            sf_alloc, sf_offline, sf_cum = sf_sys.get_allocation()
            hf_alloc = sf_alloc['Thruster'][str(tuple())]   # [Neval, total cost]
            hf_model_cost = hf_alloc[1] / hf_alloc[0]
            mf_test = mf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            mf_alloc, mf_offline, mf_cum = mf_sys.get_allocation()

            # Plot QoI L2 error on test set vs. cost
            qoi_ind = sf_sys._get_qoi_ind(qoi_ind)
            labels = [sf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
            fig, axs = plt.subplots(1, len(qoi_ind), sharey='row')
            for i in range(len(qoi_ind)):
                ax = axs[i] if len(qoi_ind) > 1 else axs
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='MF')
                ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='SF')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.grid()
                ax.set_title(labels[i])
                ylabel = r'Relative $L_2$ error' if i == 0 else ''
                ax_default(ax, r'Cost', ylabel, legend=i+1 == len(qoi_ind))
            fig.set_size_inches(3.5*len(qoi_ind), 3.5)
            fig.tight_layout()
            fig.savefig(root_dir/'error_v_cost.png', dpi=300, format='png')

            # Plot cost allocation bar chart
            mf_sys.plot_allocation()


if __name__ == '__main__':
    train_mf()
