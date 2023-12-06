import numpy as np
import sys
from pathlib import Path
import pickle
import os
import datetime
from datetime import timezone
import matplotlib.pyplot as plt
import shutil
# import dill
# from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)
# from mpi4py.futures import MPICommExecutor

sys.path.append('../..')

# Custom imports
from models.thruster import thruster_pem, uion_reconstruct
from utils import UniformRV, load_variables, ax_default
from surrogates.system import SystemSurrogate   # This is the data structure that does all the surrogate stuff


def predict_ion_velocity(x, surr=None, root_dir=None, truth=False):
    """Predict an ion velocity curve with the surrogate
    :param x: (..., 5) Any shape np.ndarray where last dimension has the inputs of [PB, z_start, z_end, vAN1, vAN2]
    :param surr: the SystemSurrogate object to use for prediction, will try to load from root_dir if None
    :param root_dir: Path/str of surrogate root build directory (overridden if surr is provided directly)
    :param truth: Also return z, uion for the ground truth comparison of the full simulation
    :returns z, uion: (202,), and (..., 202) Returns the pair (z, uion(z)) at 202 points for z=[0, 0.08] m
                                             along channel centerline, also returns zt, uion_truth if truth=True
    """
    if surr is None:
        # Find the most recent madison_timestamp build directory
        if root_dir is None:
            results_dir = Path('.')   # train_surrogate() will put all build folders in current directory
            files = [f for f in os.listdir(results_dir) if f.startswith('madison_')]
            root_dir = results_dir / files[0]

        # Load the final trained surrogate
        surr = SystemSurrogate.load_from_file(root_dir / 'sys' / 'sys_final.pkl')

    # Predict to get the ion velocity latent coefficients (and show against the true simulation
    yr_hat = surr(x)                        # (..., r1), where r1 is the number of latent coefficients (20)
    z, uion_hat = uion_reconstruct(yr_hat)  # Reconstructs to z=np.linspace(0, 0.08, 202), uion = (..., 202) in m/s

    if truth:
        yr_true = surr(x, use_model='best')
        zt, uion_truth = uion_reconstruct(yr_true)
        return z, uion_hat, zt, uion_truth

    return z, uion_hat


def train_surrogate(executor=None):
    """Train hallthruster.jl surrogate in parallel on MPI workers on Great Lakes"""
    # Set up build directory
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = 'madison_' + timestamp
    os.mkdir(root_dir)

    # Open test set
    with open(Path('../models/data') / 'test_set.pkl', 'rb') as fd:
        test_set = pickle.load(fd)  # Dict('xt': array(Nt, xdim), 'yt': array(Nt, ydim))

    # Configure and train the surrogate
    qoi_ind = [0, 1, 2]  # just first 3 latent coefficients for guiding the training process
    surr = config_surrogate(executor=executor, root_dir=root_dir)
    surr.build_system(qoi_ind=qoi_ind, N_refine=1000, max_iter=200, max_tol=1e-4, max_runtime=3,
                      save_interval=50, prune_tol=1e-8, n_jobs=-1, test_set=test_set)
    return surr


def config_surrogate(executor=None, root_dir=None, init=True):
    """Return a SystemSurrogate object for hallthruster.jl"""
    # Load surrogate input variables
    exo_vars = load_variables(['PB', 'z_start', 'z_end', 'vAN1', 'vAN2'])

    # Get number of latent coefficients for ion velocity profile
    with open(Path(__file__).parent / '..' / '..' / 'models' / 'data' / 'thruster_svd.pkl', 'rb') as fd:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        coupling_vars = [UniformRV(-20, 20, id=f'uion{i}', tex=f"$\\tilde{{u}}_{{ion,{i}}}$",
                                   description=f'Ion velocity latent coefficient {i}',
                                   param_type='coupling') for i in range(r1)]   # 'coupling' vars are outputs

    # Models must be specified at global scope
    thruster = {'name': 'Thruster', 'model': thruster_pem, 'truth_alpha': (2, 2), 'max_alpha': (2, 2),
                'exo_in': ['PB', 'z_start', 'z_end', 'vAN1', 'vAN2'], 'coupling_in': [],
                'coupling_out': [f'uion{i}' for i in range(r1)], 'type': 'lagrange', 'max_beta': (3, 3, 3, 3),
                'save_output': True, 'model_args': (), 'model_kwargs': {'n_jobs': -1, 'compress': True,
                                                                        'config': Path('hallthruster_jl.json')}}
    surr = SystemSurrogate([thruster], exo_vars, coupling_vars, executor=executor, stdout=False,
                           root_dir=root_dir, init_surr=init)

    return surr


def gen_svd_data(N=500, r_pct=0.999):
    """Generate data matrices for SVD dimension reduction"""
    # Thruster svd dataset for uion velocity profile
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path(f'svd_{timestamp}')
    os.mkdir(root_dir)
    surr = config_surrogate(executor=None, root_dir=root_dir, init=False)
    xt = surr.sample_inputs((N,), comp='Thruster', use_pdf=True)
    comp = surr['Thruster']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, use_model='best', model_dir=comp._model_kwargs.get('output_dir'))
    nan_idx = np.any(np.isnan(yt), axis=-1)
    A = yt[~nan_idx, :] # Data matrix for uion
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

    return root_dir


def gen_test_set(N=1000):
    """Generate a test set of high-fidelity model solves"""
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path(f'test_{timestamp}')
    os.mkdir(root_dir)
    surr = config_surrogate(executor=None, init=False, root_dir=root_dir)
    xt = surr.sample_inputs((N,), use_pdf=True)     # (N, xdim)
    yt = surr(xt, use_model='best', training=False, model_dir=surr['Thruster']._model_kwargs.get('output_dir'))
    nan_idx = np.any(np.isnan(yt), axis=-1)
    xt = xt[~nan_idx, :]
    yt = yt[~nan_idx, :]
    data = {'xt': xt, 'yt': yt}

    with open(Path(surr.root_dir) / 'test_set.pkl', 'wb') as dill_file:
        pickle.dump(data, dill_file)

    return root_dir


if __name__ == '__main__':
    # STEP 1) Generate SVD and test set files
    svd_dir = gen_svd_data()
    copy_dir = Path(__file__).parent / '..' / '..' / 'models' / 'data'
    shutil.copyfile(svd_dir / 'thruster_svd.pkl', copy_dir / 'thruster_svd.pkl')
    test_dir = gen_test_set()
    shutil.copyfile(test_dir / 'test_set.pkl', copy_dir / 'test_set.pkl')

    # STEP 2) Run training on Great lakes with MPI parallel execution
    # with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    #     if executor is not None:
    #         surr = train_surrogate(executor=executor)
    #
    #         # Plot some 1d slices to check how training went
    #         nominal = {str(var): var.sample_domain((1,)) for var in surr.exo_vars}  # Random nominal test point
    #         fig, ax = surr.plot_slice([0, 1, 2, 3, 4], [0, 1, 2], show_model=['best', 'worst'], show_surr=True, N=20,
    #                                   random_walk=False, nominal=nominal, model_dir=surr.root_dir)
    #         plt.show()

    # STEP 3) Test surrogate prediction of full reconstructed ion velocity field
    # PB = -5     # log10 Torr
    # zs = 0.015
    # ze = 0.035
    # vAN1 = -2   # vAN1 -> [-4, -1]
    # vAN2 = 20   # vAN2 -> [10, 100]
    # x = np.array([PB, zs, ze, vAN1, vAN2])
    # zh, uion_hat, zt, uion_truth = predict_ion_velocity(x, truth=True)
    # fig, ax = plt.subplots()
    # ax.plot(zt, uion_truth, '-k', label='Model')
    # ax.plot(zh, uion_hat, '--r', label='Surrogate')
    # ax_default(ax, 'Axial location from anode [m]', 'Ion velocity [m/s]', legend=True)
    # plt.show()
