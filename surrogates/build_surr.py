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


def gen_svd_data(N=500, r_pct=0.999):
    """Generate data matrices for SVD dimension reduction"""
    # Thruster svd dataset for uion velocity profile
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path('../results/svd') / f'svd_{timestamp}'
    os.mkdir(root_dir)
    surr = pem_system(executor=None, init=False, root_dir=root_dir)
    xt = surr.sample_inputs((N,), comp='Thruster')
    comp = surr['Thruster']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, ground_truth=True)
    nan_idx = np.any(np.isnan(yt), axis=-1)
    yt = yt[~nan_idx, :]
    A = yt[:, 6:]       # Data matrix, uion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s**2 / np.sum(s**2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1         # Number of singular values to keep
    vtr = vt[:r, :]     # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open(root_dir / 'thruster_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str(root_dir/'thruster_svd.png'), dpi=300, format='png')
    fig, ax = plt.subplots()
    M = vtr.shape[1]            # Number of grid cells
    zg = np.linspace(0, 1, M)   # Normalized grid locations
    lb = np.percentile(A, 5, axis=0)
    mid = np.percentile(A, 50, axis=0)
    ub = np.percentile(A, 95, axis=0)
    ax.plot(zg, mid, '-k')
    ax.fill_between(zg, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax_default(ax, 'Normalized axial location', 'Ion velocity (m/s)', legend=False)
    fig.savefig(str(root_dir/'uion.png'), dpi=300, format='png')

    # Generate SVD data matrix for Plume
    xt = surr.sample_inputs((N,), comp='Plume')
    comp = surr['Plume']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, ground_truth=True)
    idx = ~np.isnan(yt[:, 0]) & (np.nanmax(yt, axis=-1) <= 1000)    # Remove some outliers above 1000 A/m^2
    yt = yt[idx, :]
    A = yt[:, 1:]  # Data matrix, jion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s ** 2 / np.sum(s ** 2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1  # Number of singular values to keep
    vtr = vt[:r, :]  # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open(root_dir / 'plume_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str(root_dir / 'plume_svd.png'), dpi=300, format='png')
    fig, ax = plt.subplots()
    M = vtr.shape[1]  # Number of grid cells
    zg = np.linspace(0, 1, M)  # Normalized grid locations
    lb = np.percentile(A, 5, axis=0)
    mid = np.percentile(A, 50, axis=0)
    ub = np.percentile(A, 95, axis=0)
    ax.plot(zg, mid, '-k')
    ax.fill_between(zg, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax.set_yscale('log')
    ax_default(ax, 'Normalized angle', 'Current density ($A/m^2$)', legend=False)
    fig.savefig(str(root_dir / 'jion.png'), dpi=300, format='png')


def gen_test_set(N=1000):
    """Generate a test set of high-fidelity model solves"""
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path('../results/test') / f'test_{timestamp}'
    os.mkdir(root_dir)
    sys = pem_system(executor=None, init=False, root_dir=root_dir)
    xt = sys.sample_inputs((N,))     # (N, xdim)
    yt = sys(xt, ground_truth=True, training=False)
    nan_idx = np.any(np.isnan(yt), axis=-1)
    xt = xt[~nan_idx, :]
    yt = yt[~nan_idx, :]
    data = {'xt': xt, 'yt': yt}

    with open(Path(sys.root_dir) / 'test_set.pkl', 'wb') as dill_file:
        dill.dump(data, dill_file)


def train_mf():
    """Train and compare MF v. SF surrogates"""
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            with open(Path('../models/data') / 'test_set.pkl', 'rb') as fd:
                test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))
            qoi_ind = [1, 2, 8, 9]  # just I_B0, thrust, and first two u_ion latent coefficients
            # qoi_ind = [0, 1, 2, 7, 8, int(7 + r1), int(7 + r1 + 1), int(7 + r1 + 2)]

            # Set up multi-fidelity vs. single-fidelity comparison folders
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            root_dir = Path('../results/surrogates') / f'mf_{timestamp}'
            os.mkdir(root_dir)
            os.mkdir(root_dir / 'single-fidelity')
            os.mkdir(root_dir / 'multi-fidelity')

            # Single-fidelity build
            sf_sys = pem_system(executor=executor, init=True, root_dir=root_dir / 'single-fidelity', hf_override=True)
            sf_sys.build_system(qoi_ind=qoi_ind, N_refine=1000, max_iter=200, max_tol=1e-4, max_runtime=3,
                                save_interval=50, test_set=test_set, prune_tol=1e-8, n_jobs=-1)

            # Multi-fidelity build
            mf_sys = pem_system(executor=executor, init=True, root_dir=root_dir / 'multi-fidelity', hf_override=False)
            mf_sys.build_system(qoi_ind=qoi_ind, N_refine=1000, max_iter=200, max_tol=1e-4, max_runtime=3,
                                save_interval=50, test_set=test_set, prune_tol=1e-8, n_jobs=-1)

            # Get cost allocation for sf and mf systems
            sf_test = sf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            sf_alloc, sf_offline, sf_cum = sf_sys.get_allocation()
            hf_alloc = sf_alloc['Thruster'][str(tuple())]   # [Neval, total cost]
            hf_model_cost = hf_alloc[1] / hf_alloc[0]
            mf_test = mf_sys.build_metrics['test_stats']    # (Niter+1, 2, Nqoi)
            mf_alloc, mf_offline, mf_cum = mf_sys.get_allocation()

            # Get mf total cost (including offline overhead)
            total_cost = mf_cum[-1]
            for node, alpha_dict in mf_offline.items():
                for alpha, cost in alpha_dict.items():
                    total_cost += cost[1]

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

            # Remove nodes with cost=0 from alloc dicts (i.e. analytical models)
            remove_nodes = []
            for node, alpha_dict in mf_alloc.items():
                if len(alpha_dict) == 0:
                    remove_nodes.append(node)
            for node in remove_nodes:
                del mf_alloc[node]
                del mf_offline[node]

            # Bar chart showing cost allocation breakdown for MF system at end
            fig, axs = plt.subplots(1, 2, sharey='row')
            width = 0.8
            x = np.arange(len(mf_alloc))
            xlabels = list(mf_alloc.keys())
            cmap = plt.get_cmap('viridis')
            for k in range(2):
                ax = axs[k]
                alloc = mf_alloc if k == 0 else mf_offline
                ax.set_title('Online training' if k == 0 else 'Overhead')
                for j, (node, alpha_dict) in enumerate(alloc.items()):
                    bottom = 0
                    c_intervals = np.linspace(0, 1, len(alpha_dict))
                    for i, (alpha, cost) in enumerate(alpha_dict.items()):
                        p = ax.bar(x[j], cost[1] / total_cost, width, color=cmap(c_intervals[i]), linewidth=1,
                                   edgecolor=[0, 0, 0], bottom=bottom)
                        bottom += cost[1] / total_cost
                        ax.bar_label(p, labels=[f'{alpha} - {round(cost[0])}'], label_type='center')
                ax_default(ax, '', "Fraction of total cost" if k == 0 else '', legend=False)
                ax.set_xticks(x, xlabels)
                ax.set_xlim(left=-1, right=1)
            fig.set_size_inches(8, 4)
            fig.tight_layout()
            fig.savefig(root_dir/'mf_allocation.png', dpi=300, format='png')
            plt.show()


if __name__ == '__main__':
    gen_svd_data()
    # gen_test_set()
    # train_mf()
