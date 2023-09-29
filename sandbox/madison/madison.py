import numpy as np
import sys
import copy
from pathlib import Path
import pickle
import json
import uuid
import time
from joblib import cpu_count, Parallel, delayed
from joblib.externals.loky import set_loky_pickler
import tempfile
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
from models.thruster import hallthruster_jl_model, uion_reconstruct
from utils import UniformRV, load_variables, ModelRunException, get_logger, data_write, ax_default
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
    thruster = {'name': 'Thruster', 'model': thruster_madison, 'truth_alpha': (2, 2), 'max_alpha': (2, 2),
                'exo_in': ['PB', 'vAN1', 'vAN2'], 'coupling_in': [], 'coupling_out': [f'uion{i}' for i in range(r1)],
                'type': 'lagrange', 'max_beta': (3, 3, 3), 'save_output': True,
                'model_args': (), 'model_kwargs': {'n_jobs': -1, 'compress': True}}
    surr = SystemSurrogate([thruster], exo_vars, coupling_vars, executor=executor, suppress_stdout=True,
                           root_dir=root_dir)

    return surr


def thruster_madison(x, alpha, *args, compress=True, output_dir=None, n_jobs=-1, config='hallthruster_jl.json',
                     **kwargs):
    """Run Hallthruster.jl in Madison's format on the SPT-100
    :param x: (..., xdim) Inputs: ['PB', 'vAN1', 'vAN2']
    :param alpha: tuple(alpha_1, alpha_2) Model fidelity indices = (N_cells, N_charge)
    :param compress: Whether to compress the ion velocity profile
    :param output_dir: str or Path specifying where to save Hallthruster.jl results
    :param n_jobs: (int) number of jobs to run in parallel, use all cpus if -1
    :param config: (str) of config file to load static thruster simulation configs from models/config
    :returns dict(y=(..., ydim), files=[], cost=float), model outputs, files, and avg model cost
    """
    # Set model fidelity quantities from alpha
    Ncells = 50 * (alpha[0] + 2)
    Ncharge = alpha[1] + 1
    dt_map = [12.5e-9, 8.4e-9, 6.3e-9]
    dt_s = dt_map[alpha[0]] if Ncharge <= 2 else dt_map[alpha[0]] / np.sqrt(3/2)

    # Constant inputs from config file (SPT-100 geometry, propellant, wall material, simulation params, etc.)
    model_dir = Path(__file__).parent / '..' / '..' / 'models'
    with open(model_dir / 'config' / config, 'r') as fd:
        base_input = json.load(fd)['SPT-100']
    base_input.update({'num_cells': Ncells, 'dt_s': dt_s, 'max_charge': Ncharge})

    # Some extra constant inputs for Madison's case
    base_input.update({
        'anode_potential': 300,                 # [V]
        'anode_mass_flow_rate': 5.16 * 1e-6,    # [kg/s]
        'cathode_electron_temp_eV': 1.99,       # [eV]
        'neutral_velocity_m_s': 300,            # [m/s]
        'sheath_loss_coefficient': 0.2176,      # [-]
        'inner_outer_transition_length_m': 10.93 * 1e-3,    # [m]
        'cathode_location_m': 0.08,             # [m]
        'cathode_potential': 30                 # [V]
    })

    # Load svd params for dimension reduction of ion velocity profile
    if compress:
        with open(model_dir / 'data' / 'thruster_svd.pkl', 'rb') as fd:
            svd_data = pickle.load(fd)
            vtr = svd_data['vtr']   # (r x M)
            r, M = vtr.shape        # r-> latent (compressed) dimension, M-> original high-dimension of PDE mesh
            ydim = r
    else:
        M = Ncells + 2
        ydim = M

    # Save the inputs to file
    eval_id = str(uuid.uuid4())
    if output_dir is not None:
        save_dict = {'alpha': alpha, 'x': x}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

    # This is just a workaround for being able to run hallthruster.jl in parallel
    def run_batch(job_num, index_batches, y):
        """Run a batch of indices into the input matrix x"""
        from juliacall import Main as jl
        jl.seval('using HallThruster')
        thruster_input = copy.deepcopy(base_input)
        curr_batch = index_batches[job_num]
        files = []  # Return an ordered list of output filenames corresponding to input indices
        costs = []  # Time required to evaluate hallthruster.jl for a single input

        for i, index in enumerate(curr_batch):
            x_curr = [float(x[index + (i,)]) for i in range(x.shape[-1])]   # (xdim,)
            thruster_input.update({
                'background_pressure_Torr': 10 ** x_curr[0],
                'anom_coeff_1': 10 ** x_curr[1],
                'anom_coeff_2': x_curr[2],
            })

            # Run hallthruster.jl for one set of inputs
            t1 = time.time()
            try:
                res = hallthruster_jl_model(thruster_input, jl=jl)
            except ModelRunException as e:
                logger.warning(f'Skipping index {index} due to caught exception: {e}')
                y[index + (slice(None),)] = np.nan
                if output_dir is not None:
                    save_dict = {'input': thruster_input, 'Exception': str(e), 'index': index}
                    fname = f'{eval_id}_{index}_exc.json'
                    files.append(fname)
                    costs.append(0)
                    data_write(save_dict, fname, output_dir)
                continue

            # Save ion velocity (either in compressed or full PDE mesh form)
            if compress:
                # Interpolate ion velocity to the full reconstruction grid (of dim M)
                n_cells = M - 2     # M = number of grid points = Ncells + 2 (half-grid cells at ends of FE domain)
                L = thruster_input['cathode_location_m']    # Cathode location is the end of axial z domain
                dz = L / n_cells
                zg = np.zeros(M)    # zg is the axial z grid points for the reconstructed field (of size M)
                zg[0] = 0
                zg[1] = dz / 2
                zg[2:-1] = zg[1] + np.arange(1, n_cells) * dz
                zg[-1] = L
                z1 = np.atleast_1d(res['z'])
                ui1 = np.atleast_1d(res['ui_1'])
                uig = np.interp(zg, z1, ui1)  # Interpolated ui_1 on reconstruction grid (M,)
                y[index + (slice(None),)] = np.squeeze(vtr @ uig[..., np.newaxis], axis=-1)  # Compress to dim (r,)
            else:
                # Otherwise, save entire ion velocity grid
                y[index + (slice(None),)] = res['ui_1']
            costs.append(time.time() - t1)  # Save single model wall clock runtime in seconds (on one cpu)

            # Save to output file (delete redundant results to save space)
            if output_dir is not None:
                del res['ni_1']
                del res['ni_2']
                del res['ni_3']
                del res['grad_pe']
                del res['E']
                del res['mobility']
                if Ncharge < 3:
                    del res['ui_3']
                    del res['niui_3']
                if Ncharge < 2:
                    del res['ui_2']
                    del res['niui_2']
                save_dict = {'input': thruster_input, 'output': res}
                fname = f'{eval_id}_{index}.json'
                files.append(fname)
                data_write(save_dict, fname, output_dir)

        return files, costs

    # Evenly distribute input indices across batches
    num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
    index_batches = [list() for i in range(num_batches)]
    flat_idx = 0
    for input_index in np.ndindex(*x.shape[:-1]):
        # Cartesian product iteration over x.shape indices
        index_batches[flat_idx % num_batches].append(input_index)
        flat_idx += 1

    # Allocate space for outputs and compute model (in parallel batches)
    set_loky_pickler('cloudpickle')     # Dill can't serialize mmap objects, but cloudpickle can
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd:
        pass
    y = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=x.shape[:-1] + (ydim,))
    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        res = ppool(delayed(run_batch)(job_num, index_batches, y) for job_num in range(num_batches))
    y_ret = np.zeros(y.shape)
    y_ret[:] = y[:]
    del y
    os.unlink(y_fd.name)

    # Save model eval summary to file
    files, costs = [], []
    if output_dir is not None:
        # Re-order the resulting list of file names
        flat_idx = 0
        for input_index in np.ndindex(*x.shape[:-1]):
            # Iterate in same circular fashion as the inputs were passed to parallel
            batch_files, batch_costs = res[flat_idx % num_batches]
            files.append(batch_files.pop(0))
            costs.append(batch_costs.pop(0))
            flat_idx += 1

        save_dict = {'alpha': alpha, 'x': x, 'y': y_ret, 'is_compressed': compress, 'files': files, 'costs': costs}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)
    costs = np.atleast_1d(costs)
    costs[costs == 0] = np.nan
    avg_model_cpu_time = np.nanmean(costs)

    return {'y': y_ret, 'files': files, 'cost': avg_model_cpu_time}


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
