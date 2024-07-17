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
from amisc import IndicesRV, IndexSet
from joblib import Parallel, delayed

from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def train_mf(max_runtime_hr=16):
    """Train and compare MF v. SF surrogates."""
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Set up multi-fidelity vs. single-fidelity surrogates
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            root_dir = Path('results') / f'mf_{timestamp}'
            os.mkdir(root_dir)
            os.mkdir(root_dir / 'single-fidelity')
            os.mkdir(root_dir / 'multi-fidelity')
            # sf_sys = pem_v0(executor=executor, init=False, save_dir=root_dir / 'single-fidelity', hf_override=True)
            mf_sys = pem_v0(executor=executor, init=False, save_dir=root_dir / 'multi-fidelity', hf_override=False)
            # with open('results/mf_2024-06-12T23.50.31/multi-fidelity/amisc_2024-06-12T23.50.31/sys/sys_final.pkl', 'rb') as fd:
            #     mf_sys = pickle.load(fd)
            # with open('results/mf_2024-06-12T23.50.31/single-fidelity/amisc_2024-06-12T23.50.31/sys/sys_final.pkl', 'rb') as fd:
            #     sf_sys = pickle.load(fd)

            # Filter test set
            with open(CONFIG_DIR / 'test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            lb = np.array([var.bounds()[0] for var in mf_sys.exo_vars])
            ub = np.array([var.bounds()[1] for var in mf_sys.exo_vars])
            keep_idx = np.all((test_set['xt'] < ub) & (test_set['xt'] > lb), axis=1)
            test_set['xt'] = test_set['xt'][keep_idx, :]
            test_set['yt'] = test_set['yt'][keep_idx, :]

            # sf_sys.init_system()
            mf_sys.init_system()

            qoi_ind = ['I_D', 'T', 'uion0']
            # sf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=200, max_runtime=max_runtime_hr,
            #            save_interval=50, test_set=test_set, n_jobs=-1)
            mf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=200, max_runtime=max_runtime_hr,
                       save_interval=50, test_set=test_set, n_jobs=-1)

            # Get cost allocation for sf and mf systems
            # sf_test = sf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            # sf_alloc, sf_offline, sf_cum = sf_sys.get_allocation()
            # hf_alloc = sf_alloc['Thruster'][str(tuple())]   # [Neval, total cost]
            # hf_model_cost = hf_alloc[1] / hf_alloc[0]
            mf_test = mf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            mf_alloc, mf_offline, mf_cum = mf_sys.get_allocation()
            hf_alloc = mf_alloc['Thruster']['(0, 0)']
            hf_model_cost = hf_alloc[1] / hf_alloc[0]

            # Plot QoI L2 error on test set vs. cost
            qoi_ind = mf_sys._get_qoi_ind(qoi_ind)
            labels = [mf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
            fig, axs = plt.subplots(1, len(qoi_ind), sharey='row')
            for i in range(len(qoi_ind)):
                ax = axs[i] if len(qoi_ind) > 1 else axs
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='Multi-fidelity')
                # ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='Single-fidelity')
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


def get_test_set_error(surr, qoi_ind: IndicesRV = None, max_iter: int = 20, 
                       test_set: dict = None, n_jobs: int = 1, root_dir = '.'):
        """Simulate training surrogate to generate test set error statistics.

        :param qoi_ind: list of system QoI variables to focus refinement on, use all QoI if not specified
        :param max_iter: the maximum number of refinement steps to take
        :param test_set: `dict(xt=(Nt, x_dim), yt=(Nt, y_dim)` to show convergence of surrogate to the truth model
        :param n_jobs: number of cpu workers for computing error indicators (on master MPI task), 1=sequential
        """
        qoi_ind = surr._get_qoi_ind(qoi_ind)
        Nqoi = len(qoi_ind)
        max_iter = surr.refine_level + max_iter
        cum_cost = 0
        test_stats, xt, yt, t_fig, t_ax = None, None, None, None, None

        # Record of (error indicator, component, alpha, beta, num_evals, total added cost (s)) for each iteration
        train_record = surr.build_metrics.get('train_record', [])
        if test_set is not None:
            xt, yt = test_set['xt'], test_set['yt']
        # xt, yt = surr.build_metrics.get('xt', xt), surr.build_metrics.get('yt', yt)  # Overrides test set param

        # Track convergence progress on a test set and on the max error indicator
        err_fig, err_ax = plt.subplots()
        if xt is not None and yt is not None:
            surr.build_metrics['xt'] = xt
            surr.build_metrics['yt'] = yt
            test_stats = np.expand_dims(surr.get_test_metrics(xt, yt, qoi_ind=qoi_ind), axis=0)
            t_fig, t_ax = plt.subplots(1, Nqoi) if Nqoi > 1 else plt.subplots()

        def activate_index(alpha, beta):
            # Add all possible new candidates (distance of one unit vector away)
            ele = (alpha, beta)
            ind = list(alpha + beta)
            new_candidates = []
            for i in range(len(ind)):
                ind_new = ind.copy()
                ind_new[i] += 1

                # Don't add if we surpass a refinement limit
                if np.any(np.array(ind_new) > np.array(surr['Thruster'].max_refine)):
                    continue

                # Add the new index if it maintains downward-closedness
                new_cand = (tuple(ind_new[:len(alpha)]), tuple(ind_new[len(alpha):]))
                down_closed = True
                for j in range(len(ind)):
                    ind_check = ind_new.copy()
                    ind_check[j] -= 1
                    if ind_check[j] >= 0:
                        tup_check = (tuple(ind_check[:len(alpha)]), tuple(ind_check[len(alpha):]))
                        if tup_check not in index_set and tup_check != ele:
                            down_closed = False
                            break
                if down_closed:
                    new_candidates.append(new_cand)

            # Move to the active index set
            if ele in candidate_set:
                candidate_set.remove(ele)
            index_set.append(ele)
            new_candidates = [cand for cand in new_candidates if cand not in candidate_set]
            candidate_set.extend(new_candidates)

            # Return total cost of activation
            total_cost = 0.0
            for a, b in new_candidates:
                total_cost += surr['Thruster'].get_cost(a, b)
            return total_cost

        # Set up a parallel pool of workers, sequential if n_jobs=1
        with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
            i = 0
            error_record = []
            index_set = []
            candidate_set = []
            while True:
                # Check all end conditions
                if surr.refine_level >= max_iter:
                    surr._print_title_str(f'Termination criteria reached: Max iteration {surr.refine_level}/{max_iter}')
                    break
                if i == 75:
                    break
                print(i)

                # Plot progress of error indicator
                # train_record.append(refine_res)
                # error_record = [res[0] for res in train_record]
                # surr.build_metrics['train_record'] = train_record
                err_indicator, node, alpha, beta, num_evals, cost = surr.build_metrics['train_record'][i]
                error_record.append(err_indicator)
                new_cost = activate_index(alpha, beta)
                cum_cost += new_cost
                yt = surr.predict(xt, 100, 10, 1e-10, None, None, False, False, {node: index_set}, qoi_ind, None)

                # Plot progress on test set
                if xt is not None and yt is not None:
                    stats = surr.get_test_metrics(xt, yt, qoi_ind=[0,1,2,3])
                    test_stats = np.concatenate((test_stats, stats[np.newaxis, ...]), axis=0)
                    surr.build_metrics['test_stats'] = test_stats

                i+=1

        for i in range(Nqoi):
            ax = t_ax if Nqoi == 1 else t_ax[i]
            ax.clear(); ax.grid(); ax.set_yscale('log')
            ax.plot(test_stats[:, 1, i], '-k')
            ax.set_title(surr.coupling_vars[qoi_ind[i]].to_tex(units=True))
            ax_default(ax, 'Iteration', r'Relative $L_2$ error', legend=False)
        t_fig.set_size_inches(3.5*Nqoi, 3.5)
        t_fig.tight_layout()
        if root_dir is not None:
            t_fig.savefig(str(Path(root_dir) / 'test_set.png'), dpi=300, format='png')
        
        err_ax.clear(); err_ax.grid(); err_ax.plot(error_record, '-k')
        ax_default(err_ax, 'Iteration', r'Relative $L_2$ error indicator', legend=False)
        err_ax.set_yscale('log')
        if root_dir is not None:
                    err_fig.savefig(str(Path(root_dir) / 'error_indicator.png'), dpi=300, format='png')

        print(cum_cost)
        return test_stats


if __name__ == '__main__':
    # train_mf()
    with open('../../results/mf_2024-07-09T20.25.34/multi-fidelity/amisc_2024-07-09T20.25.35/sys/sys_final.pkl', 'rb') as fd:
        mf_sys = pickle.load(fd)
    with open(CONFIG_DIR / 'test_set.pkl', 'rb') as fd:
        test_set = pickle.load(fd)
    lb = np.array([var.bounds()[0] for var in mf_sys.exo_vars])
    ub = np.array([var.bounds()[1] for var in mf_sys.exo_vars])
    keep_idx = np.all((test_set['xt'] < ub) & (test_set['xt'] > lb), axis=1)
    test_set['xt'] = test_set['xt'][keep_idx, :]
    test_set['yt'] = test_set['yt'][keep_idx, :]

    stats = get_test_set_error(mf_sys, ['I_B0', 'I_D', 'T', 'uion0'], 75, test_set, 1, '.')
    print(stats)
    mf_test = stats    # (Niter+1, 2, Nqoi)
    mf_alloc, mf_offline, mf_cum = mf_sys.get_allocation()
    hf_alloc = mf_alloc['Thruster']['(0, 0)']
    hf_model_cost = hf_alloc[1] / hf_alloc[0]

    # Plot QoI L2 error on test set vs. cost
    qoi_ind = mf_sys._get_qoi_ind(['I_B0', 'I_D', 'T', 'uion0'])
    labels = [mf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
    fig, axs = plt.subplots(1, len(qoi_ind), sharey='row')
    for i in range(len(qoi_ind)):
        ax = axs[i] if len(qoi_ind) > 1 else axs
        ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='Multi-fidelity')
        # ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='Single-fidelity')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.set_title(labels[i])
        ylabel = r'Relative $L_2$ error' if i == 0 else ''
        ax_default(ax, r'Cost', ylabel, legend=i+1 == len(qoi_ind))
    fig.set_size_inches(3.5*len(qoi_ind), 3.5)
    fig.tight_layout()
    fig.savefig('error_v_cost.png', dpi=300, format='png')
