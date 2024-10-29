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
import time
import dill
from mpi4py import MPI
MPI.pickle.__init__(dill.dumps, dill.loads)
from mpi4py.futures import MPICommExecutor
import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default
from amisc import IndicesRV, IndexSet
from joblib import Parallel, delayed
from amisc.system import SystemSurrogate
from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir
CONFIG_DIR = model_config_dir()


def train_mf(max_runtime_hr=12):
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

            sf_sys.init_system()
            mf_sys.init_system()
            qoi_ind = ['I_D', 'T', 'uion0']
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
            #hf_alloc = mf_alloc['Thruster']['(0, 0)']
            #hf_model_cost = hf_alloc[1] / hf_alloc[0]

            # Plot QoI L2 error on test set vs. cost
            qoi_ind = sf_sys._get_qoi_ind(qoi_ind)
            labels = [sf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
            fig, axs = plt.subplots(1, len(qoi_ind), sharey='row')
            for i in range(len(qoi_ind)):
                ax = axs[i] if len(qoi_ind) > 1 else axs
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='Multi-fidelity')
                ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='Single-fidelity')
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


def get_test_set_error(surr, qoi_ind):
    """Simulate training surrogate to generate test set error statistics.

    :param surr: the SystemSurrogate object
    :param qoi_ind: list of system QoI variables to focus refinement on, use all QoI if not specified
    :returns: `cost_cum`, `test_error` -- the cumulative training cost and test set error on given qois during training
    """
    qoi_ind = surr._get_qoi_ind(qoi_ind)
    Nqoi = len(qoi_ind)
    max_iter = surr.refine_level  # Iterate up to last surrogate refinement level
    xt, yt = surr.build_metrics['xt'], surr.build_metrics['yt']  # Same as the ones in test_set.pkl
    comp = surr['Thruster'] # TODO: should generalize this to arbitrary components and put in convenience amisc function
    index_set = []          # The active index set for Thruster component
    candidate_set = []      # The candidate indices for Thruster component

    def relative_l2(pred, targ, axis=-1):
        with np.errstate(divide='ignore', invalid='ignore'):
            pred, targ = np.atleast_1d(pred), np.atleast_1d(targ)
            err = np.sqrt(np.mean((pred - targ) ** 2, axis=axis) / np.mean(targ ** 2, axis=axis))
            err = np.nan_to_num(err, posinf=np.nan, neginf=np.nan, nan=np.nan)
            return err

    def activate_index(alpha, beta):
        # Add all possible new candidates (distance of one unit vector away)
        ele = (alpha, beta)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(comp.max_refine)):
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
            total_cost += comp.get_cost(a, b)
        return total_cost

    cost_cum = np.empty(max_iter+1)                # Cumulative cost allocation during training
    test_error = np.empty((max_iter+1, Nqoi))      # Relative L2 error on test set qois
    dt = []                                        # Time per surrogate prediction
    num_active = []                                # Number of active indices
    num_cand = []                                  # Number of candidate indices

    # Initial cost and test set error
    base_alpha = (0,) * len(comp.truth_alpha)
    base_beta = (0,) * (len(comp.max_refine) - len(comp.truth_alpha))
    base_cost = activate_index(base_alpha, base_beta) + comp.get_cost(base_alpha, base_beta)
    cost_cum[0] = base_cost
    t1 = time.time()
    ysurr = surr.predict(xt, index_set={'Thruster': index_set})
    dt.append(time.time() - t1)
    num_active.append(len(index_set))
    num_cand.append(len(candidate_set))
    test_error[0, :] = relative_l2(ysurr[:, qoi_ind], yt[:, qoi_ind], axis=0)

    # Iterate back over each step of surrogate training history and compute test set error
    t1 = time.time()
    for i in range(max_iter):
        err_indicator, node, alpha, beta, num_evals, cost = surr.build_metrics['train_record'][i]
        new_cost = activate_index(alpha, beta)  # Updates index_set with (alpha, beta) for Thruster node
        cost_cum[i+1] = new_cost
        t2 = time.time()
        ysurr = surr.predict(xt, index_set={'Thruster': index_set})  # (Ntest, Nqoi)
        t3 = time.time()
        dt.append(t3-t2)
        num_active.append(len(index_set))
        num_cand.append(len(candidate_set))
        test_error[i+1, :] = relative_l2(ysurr[:, qoi_ind], yt[:, qoi_ind], axis=0)
        print(f'i={i}, time={(t3-t1)/60.0:.2f} min, dt={(t3-t2)/60.0:.2f} min')

    return np.cumsum(cost_cum), test_error, dt, num_active, num_cand


def get_slice_plots(surr, qoi_ind, slice_idx):
    """Simulate training surrogate to generate plot slices during training.

    :param surr: the SystemSurrogate object
    :param qoi_ind: list of system QoI variables to make plot_slices on
    :param slice_idx: list of parameters to make slice plots with
    """
    qoi_ind = surr._get_qoi_ind(qoi_ind)
    Nqoi = len(qoi_ind)
    max_iter = surr.refine_level  # Iterate up to last surrogate refinement level
    xt, yt = surr.build_metrics['xt'], surr.build_metrics['yt']  # Same as the ones in test_set.pkl
    comp = surr['Thruster'] # TODO: should generalize this to arbitrary components and put in convenience amisc function
    index_set = []          # The active index set for Thruster component
    candidate_set = []      # The candidate indices for Thruster component

    def activate_index(alpha, beta):
        # Add all possible new candidates (distance of one unit vector away)
        ele = (alpha, beta)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(comp.max_refine)):
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
            total_cost += comp.get_cost(a, b)
        return total_cost

    num_active = []                                # Number of active indices
    num_cand = []                                  # Number of candidate indices

    # Initial cost and test set error
    base_alpha = (0,) * len(comp.truth_alpha)
    base_beta = (0,) * (len(comp.max_refine) - len(comp.truth_alpha))
    activate_index(base_alpha, base_beta)
    nominal = {str(var): var.sample_domain((1,)) for var in surr.exo_vars}  # Random nominal test point
    N = 25
    exo_bds = [var.bounds() for var in surr.exo_vars]
    index_sliceidx = [surr.exo_vars.index(var) for var in slice_idx]
    xs = np.zeros((N, len(index_sliceidx), len(surr.exo_vars)))
    for i in range(len(index_sliceidx)):
        # Make a random straight-line walk across d-cube
        r0 = np.squeeze(surr.sample_inputs((1,), use_pdf=False), axis=0)
        r0[index_sliceidx[i]] = exo_bds[index_sliceidx[i]][0]             # Start slice at this lower bound
        rf = np.squeeze(surr.sample_inputs((1,), use_pdf=False), axis=0)
        rf[index_sliceidx[i]] = exo_bds[index_sliceidx[i]][1]             # Slice up to this upper bound
        xs[0, i, :] = r0
        for k in range(1, N):
            xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
    ys_surr = surr.predict(xs, index_set={'Thruster': index_set})
    ys_model = list()
    for model in ['best', 'worst']:
        output_dir = None
        ys_model.append(surr(xs, use_model=model, model_dir=output_dir))
    save_dict = {'slice_idx': slice_idx, 'qoi_idx': qoi_idx, 'show_model': ['best', 'worst'], 'show_surr': True,
                    'nominal': nominal, 'random_walk': True, 'xs': xs, 'ys_model': ys_model, 'ys_surr': ys_surr}
    fname = 'temp.pkl'
    with open(fname, 'wb') as fd:
        pickle.dump(save_dict, fd)
    fig, axs = surr.plot_slice(slice_idx, qoi_ind, show_model=['best', 'worst'], show_surr=True, N=25,
                                    random_walk=True, nominal=nominal, model_dir=None, from_file=fname)
    fig.savefig(f'plot_slice_initial.png', format='png', bbox_inches='tight')
    num_active.append(len(index_set))
    num_cand.append(len(candidate_set))

    # Iterate back over each step of surrogate training history and compute test set error
    for i in range(max_iter): # for i in range(max_iter):
        err_indicator, node, alpha, beta, num_evals, cost = surr.build_metrics['train_record'][i]
        print(f"{i}: Alpha = ", alpha, "Beta =", beta)
        activate_index(alpha, beta)  # Updates index_set with (alpha, beta) for Thruster node
        if i % 15 == 0: # plot slice every 15 iterations
            index_sliceidx = [surr.exo_vars.index(var) for var in slice_idx]
            xs = np.zeros((N, len(index_sliceidx), len(surr.exo_vars)))
            for i in range(len(index_sliceidx)):
                # Make a random straight-line walk across d-cube
                r0 = np.squeeze(surr.sample_inputs((1,), use_pdf=False), axis=0)
                r0[index_sliceidx[i]] = exo_bds[index_sliceidx[i]][0]             # Start slice at this lower bound
                rf = np.squeeze(surr.sample_inputs((1,), use_pdf=False), axis=0)
                rf[index_sliceidx[i]] = exo_bds[index_sliceidx[i]][1]             # Slice up to this upper bound
                xs[0, i, :] = r0
                for k in range(1, N):
                    xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
            ys_surr = surr.predict(xs, index_set={'Thruster': index_set})
            ys_surr = surr.predict(xs, index_set={'Thruster': index_set})
            ys_model = list()
            for model in ['best', 'worst']:
                output_dir = None
                ys_model.append(surr(xs, use_model=model, model_dir=output_dir))
            save_dict = {'slice_idx': slice_idx, 'qoi_idx': qoi_idx, 'show_model': ['best', 'worst'], 'show_surr': True,
                            'nominal': nominal, 'random_walk': True, 'xs': xs, 'ys_model': ys_model, 'ys_surr': ys_surr}
            fname = 'temp.pkl'
            with open(fname, 'wb') as fd:
                pickle.dump(save_dict, fd)
            fig, axs = surr.plot_slice(slice_idx, qoi_ind, show_model=['best', 'worst'], show_surr=True, N=25,
                                            random_walk=True, nominal=nominal, model_dir=None, from_file=fname)
            fig.savefig(f'plot_slice_{i}.png', format='png', bbox_inches='tight')
        num_active.append(len(index_set))
        num_cand.append(len(candidate_set))
    return num_active, num_cand


def plot_test_set_error():
    """Simulate surrogate training and plot test set error v. cost"""
    mf_sys = SystemSurrogate.load_from_file('sys_final.pkl')
    qois = ['I_B0', 'I_D', 'T', 'uion0']
    cost_cum, test_error, surr_time, num_active, num_cand = get_test_set_error(mf_sys, qois)  # (Niter+1,) and (Niter+1, Nqoi)
    with open('test_set_error.pkl', 'wb') as fd:
        pickle.dump({'cost_cum': cost_cum, 'test_error': test_error, 'surr_time': surr_time,
                     'num_active': num_active, 'num_cand': num_cand}, fd)
    with open('test_set_error.pkl', 'rb') as fd:
        # Load test set results from pkl (so you don't have to run it everytime)
        data = pickle.load(fd)
        cost_cum, test_error = data['cost_cum'], data['test_error']
        surr_time, num_active, num_cand = data['surr_time'], data['num_active'], data['num_cand']
    surr_time, num_active, num_cand = map(np.array, (surr_time, num_active, num_cand))

    mf_alloc, _, _ = mf_sys.get_allocation()
    lf_alloc = mf_alloc['Thruster'][str((0,) * len(mf_sys['Thruster'].truth_alpha))]  # [Neval, total cost] for LF model
    lf_model_cost = lf_alloc[1] / lf_alloc[0]  # LF single model eval cost
    # Have to use LF cost here since the HF alpha=(2,2) wasn't reached by the mf_sys training

    # Plot QoI L2 error on test set vs. cost
    qoi_ind = mf_sys._get_qoi_ind(qois)
    old_qoi = ['I_D', 'T', 'uion0']
    labels = [mf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
    with plt.style.context('uqtils.default'):
        # Plot test set error results
        fig, ax = plt.subplots(1, len(qoi_ind), sharey='row', layout='tight', figsize=(3.5 * len(qoi_ind), 4))
        for i in range(len(qoi_ind)):
            ax[i].plot(cost_cum / lf_model_cost, test_error[:, i], '-k', label='Simulated training')
            if qois[i] in old_qoi:
                # Check that we are computing the same test set error as before
                old_test_error = mf_sys.build_metrics['test_stats'][:, 1, old_qoi.index(qois[i])]  # (Niter+1)
                ax[i].plot(cost_cum / lf_model_cost, old_test_error, '--r', label='Original training')
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].set_title(labels[i])
            ylabel = r'Relative $L_2$ error' if i == 0 else ''
            ax_default(ax[i], r'Cost (number of LF evaluations)', ylabel, legend=i == len(qoi_ind) - 1)
        fig.savefig('error_v_cost.png', dpi=200, format='png')

        # Plot surrogate evaluation time on test set against total number of candidate indices
        # TODO: find what fraction of surrogate cost is mostly spent recomputing misc coefficients
        fig, ax = plt.subplots(1, 2, layout='tight', figsize=(11, 5))
        ax[0].plot(num_active + num_cand, '-k')
        ax[1].plot(num_active + num_cand, surr_time, '-k')
        ax_default(ax[0], 'Training iteration', 'Total number of indices')
        ax_default(ax[1], 'Total number of indices', 'Surrogate evaluation time (s)')
        fig.savefig('surrogate_time.png', dpi=200, format='png')


def continue_mf(max_runtime_hr=16):
    """Train and compare MF v. SF surrogates."""
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            root_dir = '.'
            with open('../../results/mf_2024-09-21T02.15.55/multi-fidelity/amisc_2024-09-21T02.15.55/sys/sys_final.pkl', 'rb') as fd:
                mf_sys = pickle.load(fd)
            with open('../../results/mf_2024-09-21T02.15.55/single-fidelity/amisc_2024-09-21T02.15.55/sys/sys_final.pkl', 'rb') as fd:
                sf_sys = pickle.load(fd)

            # Filter test set
            with open(CONFIG_DIR / 'test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            lb = np.array([var.bounds()[0] for var in mf_sys.exo_vars])
            ub = np.array([var.bounds()[1] for var in mf_sys.exo_vars])
            keep_idx = np.all((test_set['xt'] < ub) & (test_set['xt'] > lb), axis=1)
            test_set['xt'] = test_set['xt'][keep_idx, :]
            test_set['yt'] = test_set['yt'][keep_idx, :]

            qoi_ind = ['I_D', 'T', 'uion0']
            # sf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=200, max_runtime=max_runtime_hr,
            #            save_interval=50, test_set=test_set, n_jobs=-1)
            mf_sys.fit(qoi_ind=qoi_ind, num_refine=1000, max_iter=300, max_runtime=max_runtime_hr,
                       save_interval=50, test_set=test_set, n_jobs=-1)

            # Get cost allocation for sf and mf systems
            sf_test = sf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            sf_alloc, sf_offline, sf_cum = sf_sys.get_allocation()
            hf_alloc = sf_alloc['Thruster'][str(tuple())]   # [Neval, total cost]
            hf_model_cost = hf_alloc[1] / hf_alloc[0]
            mf_test = mf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            mf_alloc, mf_offline, mf_cum = mf_sys.get_allocation()
            #hf_alloc = mf_alloc['Thruster']['(0, 0)']
            #hf_model_cost = hf_alloc[1] / hf_alloc[0]

            # Plot QoI L2 error on test set vs. cost
            qoi_ind = sf_sys._get_qoi_ind(qoi_ind)
            labels = [sf_sys.coupling_vars[idx].to_tex(units=True) for idx in qoi_ind]
            fig, axs = plt.subplots(1, len(qoi_ind), sharey='row')
            for i in range(len(qoi_ind)):
                ax = axs[i] if len(qoi_ind) > 1 else axs
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='Multi-fidelity')
                ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='Single-fidelity')
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
    # train_mf()
    # plot_test_set_error()
    # continue_mf()
    mf_sys = SystemSurrogate.load_from_file(CONFIG_DIR / 'sys_final.pkl')
    slice_idx = ['u_n', 'f_n', 'vAN2', 'vAN4', 'z0']
    qoi_idx = ['I_D', 'T', 'uion0', 'uion1']
    get_slice_plots(mf_sys, qoi_idx, slice_idx)
