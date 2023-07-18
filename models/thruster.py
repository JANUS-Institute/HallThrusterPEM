"""Module for thruster models"""
import sys
from pathlib import Path
import numpy as np
import math
import json
import tempfile
import os
import juliacall
import time
import pickle
import copy
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler
from scipy.interpolate import interp1d
import uuid

Q_E = 1.602176634e-19   # Fundamental charge (C)
sys.path.append('..')

from utils import ModelRunException, data_write, parse_input_file, get_logger
logger = get_logger(__name__)


def hallthruster_jl_input(thruster_input):
    # Format inputs for Hallthruster.jl
    json_data = dict()
    data_dir = Path('../data/spt100')
    mag_offset = thruster_input['anom_coeff_2_mag_offset']
    # vAN2 = thruster_input['anom_coeff_1'] * max(1, 10 ** mag_offset)
    vAN2 = thruster_input['anom_coeff_1'] * mag_offset
    json_data['parameters'] = {'neutral_temp_K': thruster_input['neutral_temp_K'],
                               'neutral_velocity_m_s': thruster_input['neutral_velocity_m_s'],
                               'ion_temp_K': thruster_input['ion_temp_K'],
                               'cathode_electron_temp_eV': thruster_input['cathode_electron_temp_eV'],
                               'sheath_loss_coefficient': thruster_input['sheath_loss_coefficient'],
                               'inner_outer_transition_length_m': thruster_input['inner_outer_transition_length_m'],
                               'anom_model_coeffs': [thruster_input['anom_coeff_1'], vAN2],
                               'background_pressure_Torr': thruster_input['background_pressure_Torr'],
                               'background_temperature_K': thruster_input['background_temperature_K'],
                               }
    json_data['design'] = {'thruster_name': thruster_input['thruster_name'],
                           'inner_radius': thruster_input['inner_radius'],
                           'outer_radius': thruster_input['outer_radius'],
                           'channel_length': thruster_input['channel_length'],
                           'magnetic_field_file': str(data_dir / thruster_input['magnetic_field_file']),
                           'wall_material': thruster_input['wall_material'],
                           'magnetically_shielded': thruster_input['magnetically_shielded'],
                           'anode_potential': thruster_input['anode_potential'],
                           'cathode_potential': thruster_input['cathode_potential'],
                           'anode_mass_flow_rate': thruster_input['anode_mass_flow_rate'],
                           'propellant': thruster_input['propellant_material'],
                           }
    json_data['simulation'] = {'num_cells': thruster_input['num_cells'],
                               'dt_s': thruster_input['dt_s'],
                               'duration_s': thruster_input['duration_s'],
                               'num_save': thruster_input['num_save'],
                               'cathode_location_m': thruster_input['cathode_location_m'],
                               'max_charge': thruster_input['max_charge'],
                               'flux_function': thruster_input['flux_function'],
                               'limiter': thruster_input['limiter'],
                               'reconstruct': thruster_input['reconstruct'],
                               'ion_wall_losses': thruster_input['ion_wall_losses'],
                               'electron_ion_collisions': thruster_input['electron_ion_collisions'],
                               'anom_model': thruster_input['anom_model'],
                               'solve_background_neutrals': thruster_input['solve_background_neutrals']
                               }

    # data_write(json_data, 'julia_input.json')
    return json_data


def hall_thruster_jl_model(thruster_input, jl=None):
    # Import Julia
    if jl is None:
        from juliacall import Main as jl
        jl.seval("using HallThruster")

    # Format inputs for Hallthruster.jl
    json_data = hallthruster_jl_input(thruster_input)

    # Run simulation
    try:
        fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
        json.dump(json_data, fd, ensure_ascii=False, indent=4)
        fd.close()

        # Throw a warning if transition length is too high
        l_z = json_data['parameters']['inner_outer_transition_length_m']
        if l_z < 0.001 or l_z > 0.02:
            logger.warning(f'Transition length l_z = {l_z} m is out of bounds [1mm, 20mm]')

        t1 = time.time()
        sol = jl.HallThruster.run_simulation(fd.name, verbose=False)
        # logger.info(f'Hallthruster.jl runtime: {time.time() - t1:.2f} s')
        os.unlink(fd.name)   # delete the tempfile
    except juliacall.JuliaError as e:
        raise ModelRunException(f"Julicall error in Hallthruster.jl: {e}")

    if str(sol.retcode).lower() != "success":
        raise ModelRunException(f"Exception in Hallthruster.jl: Retcode = {sol.retcode}")

    # Load simulation results
    fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
    fd.close()
    jl.HallThruster.write_to_json(fd.name, jl.HallThruster.time_average(sol, thruster_input['time_avg_frame_start']))
    with open(fd.name, 'r') as f:
        thruster_output = json.load(f)
    os.unlink(fd.name)  # delete the tempfile

    j_exit = 0      # Current density at thruster exit
    ui_exit = 0     # Ion velocity at thruster exit
    for param, grid_sol in thruster_output[0].items():
        if 'niui' in param:
            charge_num = int(param.split('_')[1])
            j_exit += Q_E * charge_num * grid_sol[-1]
        if param.split('_')[0] == 'ui':
            ui_exit += grid_sol[-1]

    A = math.pi * (thruster_input['outer_radius'] ** 2 - thruster_input['inner_radius'] ** 2)
    ui_avg = ui_exit / thruster_input['max_charge']
    I_B0 = j_exit * A           # Total current (A) at thruster exit

    thruster_output[0].update({'avg_ion_velocity': ui_avg, 'I_B0': I_B0})

    return thruster_output[0]


def thruster_pem(x, alpha, *args, compress=True, output_dir=None, n_jobs=-1, **kwargs):
    """Run Hallthruster.jl in PEM format
    :param x: (..., xdim) Thruster inputs
    :param alpha: tuple(alpha_1, alpha_2) Model fidelity indices = (N_cells, N_charge)
    :param compress: Whether to compress the ion velocity profile
    :param output_dir: str or Path specifying where to save Hallthruster.jl results
    :param n_jobs: (int) number of jobs to run in parallel, use all cpus if -1
    :returns y, files: (..., ydim) Thruster outputs and output files if output_dir is specified, otherwise just y
    """
    # Set model fidelity quantities from alpha
    Ncells = 50 * (alpha[0] + 2)
    Ncharge = alpha[1] + 1
    # dt_map = [25e-9, 12.5e-9, 8.4e-9, 6.3e-9]
    dt_map = [12.5e-9, 8.4e-9, 6.3e-9]
    dt_s = dt_map[alpha[0]] if Ncharge <= 2 else dt_map[alpha[0]] / math.sqrt(3/2)

    # Constant inputs from input file (SPT-100 geometry, propellant, wall material, simulation params, etc.)
    base_input, _ = parse_input_file('thruster_input.json')
    base_input.update({'num_cells': Ncells, 'dt_s': dt_s, 'max_charge': Ncharge})

    # Load svd params for dimension reduction
    if compress:
        with open(Path(__file__).parent / 'thruster_svd.pkl', 'rb') as fd:
            svd_data = pickle.load(fd)
            vtr = svd_data['vtr']  # (r x M)
            r, M = vtr.shape
            ydim = r + 6
    else:
        M = Ncells + 2
        ydim = M + 6

    # Save the inputs to file
    eval_id = str(uuid.uuid4())
    if output_dir is not None:
        save_dict = {'alpha': alpha, 'x': x}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

    def run_batch(job_num, index_batches, y):
        """Run a batch of indices into the input matrix x"""
        from juliacall import Main as jl
        jl.seval('using HallThruster')
        thruster_input = copy.deepcopy(base_input)
        curr_batch = index_batches[job_num]
        files = []  # Return an ordered list of output filenames corresponding to input indices

        for i, index in enumerate(curr_batch):
            x_curr = [float(x[index + (i,)]) for i in range(x.shape[-1])]   # (xdim,)
            thruster_input.update({
                'background_pressure_Torr': 10 ** x_curr[0],
                'anode_potential': x_curr[1],
                'anode_mass_flow_rate': x_curr[2] * 1e-6,
                'cathode_electron_temp_eV': x_curr[3],
                'neutral_velocity_m_s': x_curr[4],
                'sheath_loss_coefficient': x_curr[5],
                'inner_outer_transition_length_m': x_curr[6] * 1e-3,
                'anom_coeff_1': 10 ** x_curr[7],
                'anom_coeff_2_mag_offset': x_curr[8],
                'cathode_location_m': x_curr[9],
                'ion_temp_K': x_curr[10],
                'neutral_temp_K': x_curr[11],
                'background_temperature_K': x_curr[12],
                'cathode_potential': x_curr[13]
            })

            # Run hallthruster.jl
            try:
                res = hall_thruster_jl_model(thruster_input, jl=jl)
            except ModelRunException as e:
                logger.warning(f'Skipping index {index} due to caught exception: {e}')
                y[index + (slice(None),)] = np.nan
                if output_dir is not None:
                    save_dict = {'input': thruster_input, 'Exception': str(e), 'index': index}
                    fname = f'{eval_id}_{index}_exc.json'
                    files.append(fname)
                    data_write(save_dict, fname, output_dir)
                continue

            # Save QoIs
            y[index + (0,)] = res['I_B0']
            y[index + (1,)] = res['thrust']
            y[index + (2,)] = res['voltage_eff']
            y[index + (3,)] = res['current_eff']
            y[index + (4,)] = res['mass_eff']
            y[index + (5,)] = res['avg_ion_velocity']

            if compress:
                # Interpolate ion velocity to the full reconstruction grid (of dim M)
                n_cells = M - 2         # M = number of grid points = Ncells + 2 (half-grid cells at ends of FE domain)
                L = x_curr[9]           # Cathode location is the end of axial z domain
                dz = L / n_cells
                zg = np.zeros(M)    # zg is the axial z grid points for the reconstructed field (of size M)
                zg[0] = 0
                zg[1] = dz / 2
                zg[2:-1] = zg[1] + np.arange(1, n_cells) * dz
                zg[-1] = L
                z1 = np.atleast_1d(res['z'])
                ui1 = np.atleast_1d(res['ui_1'])
                uig = np.interp(zg, z1, ui1)  # Interpolated ui on reconstruction grid (M,)
                y[index + (slice(6, None),)] = np.squeeze(vtr @ uig[..., np.newaxis], axis=-1)  # Compress to dim (r,)
            else:
                # Otherwise, save entire ion velocity grid
                y[index + (slice(6, None),)] = res['ui_1']

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

        return files

    # Evenly distribute input indices across batches
    num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
    index_batches = [list() for i in range(num_batches)]
    flat_idx = 0
    for input_index in np.ndindex(*x.shape[:-1]):
        # Cartesian product iteration over x.shape indices
        index_batches[flat_idx % num_batches].append(input_index)
        flat_idx += 1

    # Allocate space for outputs and compute model (in parallel batches)
    set_loky_pickler('cloudpickle')
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
    files = []
    if output_dir is not None:
        # Re-order the resulting list of file names
        flat_idx = 0
        for input_index in np.ndindex(*x.shape[:-1]):
            # Iterate in same circular fashion as the inputs were passed to parallel
            files.append(res[flat_idx % num_batches].pop(0))
            flat_idx += 1

        save_dict = {'alpha': alpha, 'x': x, 'y': y_ret, 'is_compressed': compress, 'files': files}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

    return (y_ret, files) if output_dir is not None else y_ret


def uion_reconstruct(xr, z=None, L=0.08):
    """Reconstruct an ion velocity profile, interpolate to z if provided
    :param xr: (... r) The reduced dimension output of thruster_pem (just the ion velocity profile)
    :param z: (Nz,) The axial z grid points to interpolate to (in m, between 0 and L)
    :param L: The full domain length of the reconstructed grid
    :returns z, uion_interp: (..., Nz or M) The reconstructed (and potentially interpolated) uion profile(s),
                corresponds to z=(0, 0.08) m with M=202 points by default
    """
    with open(Path(__file__).parent / 'thruster_svd.pkl', 'rb') as fd:
        svd_data = pickle.load(fd)
        vtr = svd_data['vtr']       # (r x M)
        r, M = vtr.shape
    n_cells = M - 2
    dz = L / n_cells
    zg = np.zeros(M)  # zg is the axial z grid points for the reconstructed field (of size M)
    zg[0] = 0
    zg[1] = dz / 2
    zg[2:-1] = zg[1] + np.arange(1, n_cells) * dz
    zg[-1] = L
    uion_g = np.squeeze(vtr.T @ xr[..., np.newaxis], axis=-1)  # (..., M)

    # Do interpolation
    if z is not None:
        f = interp1d(zg, uion_g, axis=-1)
        uion_interp = f(z)  # (..., Nz)
        return z, uion_interp
    else:
        return zg, uion_g
