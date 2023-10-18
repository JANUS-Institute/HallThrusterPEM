import numpy as np
import sys
from pathlib import Path
import pickle
import os
import datetime
from datetime import timezone
import matplotlib.pyplot as plt
# import dill
# from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)
# from mpi4py.futures import MPICommExecutor

sys.path.append('../..')

# Custom imports
from models.thruster import thruster_pem, uion_reconstruct
from utils import UniformRV, load_variables, get_logger, ax_default
from surrogates.system import SystemSurrogate   # This is the data structure that does all the surrogate stuff

logger = get_logger(__name__)


def predict_ion_velocity(x, surr=None, root_dir=None, truth=False):
    """Predict an ion velocity curve with the surrogate
    :param x: (..., 3) Any shape np.ndarray where last dimension has the inputs of [PB, vAN1, vAN2]
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
        yr_true = surr(x, ground_truth=True)
        zt, uion_truth = uion_reconstruct(yr_true)
        return z, uion_hat, zt, uion_truth

    return z, uion_hat


def train_surrogate(executor=None):
    """Train hallthruster.jl surrogate in parallel on MPI workers on Great Lakes"""
    # Set up build directory
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = 'madison_' + timestamp
    os.mkdir(root_dir)

    # Configure and train the surrogate
    qoi_ind = [0, 1, 2]  # just first 3 latent coefficients for guiding the training process
    surr = config_surrogate(executor=executor, root_dir=root_dir)
    surr.build_system(qoi_ind=qoi_ind, N_refine=1000, max_iter=200, max_tol=1e-4, max_runtime=3,
                      save_interval=50, prune_tol=1e-8, n_jobs=-1)
    return surr


def config_surrogate(executor=None, root_dir=None):
    """Return a SystemSurrogate object for hallthruster.jl"""
    # Load surrogate input variables
    exo_vars = load_variables(['PB', 'vAN1', 'vAN2'])

    # Get number of latent coefficients for ion velocity profile
    with open(Path(__file__).parent / '..' / '..' / 'models' / 'data' / 'thruster_svd.pkl', 'rb') as fd:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        coupling_vars = [UniformRV(-20, 20, id=f'uion{i}', tex=f"$\\tilde{{u}}_{{ion,{i}}}$",
                                   description=f'Ion velocity latent coefficient {i}',
                                   param_type='coupling') for i in range(r1)]   # 'coupling' vars are outputs

    # Models must be specified at global scope
    thruster = {'name': 'Thruster', 'model': thruster_pem, 'truth_alpha': (2, 2), 'max_alpha': (2, 2),
                'exo_in': ['PB', 'vAN1', 'vAN2'], 'coupling_in': [], 'coupling_out': [f'uion{i}' for i in range(r1)],
                'type': 'lagrange', 'max_beta': (3, 3, 3), 'save_output': True, 'model_args': (),
                'model_kwargs': {'n_jobs': -1, 'compress': True, 'config': Path('hallthruster_jl.json')}}
    surr = SystemSurrogate([thruster], exo_vars, coupling_vars, executor=executor, suppress_stdout=True,
                           root_dir=root_dir)

    return surr


if __name__ == '__main__':
    # Run training on Great lakes with MPI parallel execution
    # with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    #     if executor is not None:
    #         surr = train_surrogate(executor=executor)
    #
    #         # Plot some 1d slices to check how training went
    #         surr.set_output_dir({'Thruster': None})  # Don't save outputs for testing
    #         surr.plot_slice([0, 1, 2], [0, 1, 2], compare_truth=True)

    # Test surrogate prediction of full reconstructed ion velocity field
    PB = -5     # log10 torr
    vAN1 = -2   # vAN1 -> [-3, -1]
    vAN2 = 20   # vAN2 -> [10, 100]
    x = np.array([PB, vAN1, vAN2])
    zh, uion_hat, zt, uion_truth = predict_ion_velocity(x, truth=True)
    fig, ax = plt.subplots()
    ax.plot(zt, uion_truth, '-k', label='Model')
    ax.plot(zh, uion_hat, '--r', label='Surrogate')
    ax_default(ax, 'Axial location from anode [m]', 'Ion velocity [m/s]', legend=True)
    plt.show()
