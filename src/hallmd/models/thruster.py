""" `thruster.py`

Module for Hall thruster models.

!!! Note
    Only current implementation is for the 1d fluid [Hallthruster.jl code](https://github.com/UM-PEPL/HallThruster.jl).
    Other thruster codes can be implemented similarly here. Place extra resources needed by the model in the `config`
    directory.

Includes
--------
- `hallthruster_jl_input()` - Used to format inputs for Hallthruster.jl
- `hallthruster_jl_model()` - Used to run Hallthruster.jl for a single set of inputs
- `hallthruster_jl_wrapper()` - The main wrapper function that is compatible with `SystemSurrogate`
- `uion_reconstruct()` - Convenience function for reconstructing ion velocity profiles from compressed data
"""
from pathlib import Path
import time
import pickle
import copy
import json
import tempfile
import os
import random
import string

import juliacall
import numpy as np
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler
from amisc.utils import load_variables, get_logger

from hallmd.utils import ModelRunException, data_write, model_config_dir

Q_E = 1.602176634e-19   # Fundamental charge (C)
CONFIG_DIR = model_config_dir()


def hallthruster_jl_input(thruster_input: dict) -> dict:
    """Format inputs for Hallthruster.jl.

    :param thruster_input: dictionary with all named thruster inputs and values
    :returns: a nested `dict` in the format that Hallthruster.jl expects to be called
    """
    anom_model = thruster_input['anom_model']
    anom_model_coeffs = []
    if anom_model == "ShiftedTwoZoneBohm" or anom_model == "TwoZoneBohm":
        vAN1 = 10 ** thruster_input['vAN1']
        vAN2 = vAN1 * thruster_input['vAN2']
        anom_model_coeffs = [vAN1, vAN2]
    elif anom_model == "ShiftedGaussianBohm":
        vAN1 = 10 ** thruster_input['vAN1']
        vAN2 = vAN1 * thruster_input['vAN2']
        vAN3 = thruster_input['vAN3']
        vAN4 = thruster_input['vAN4']
        anom_model_coeffs = [vAN1, vAN2, vAN3, vAN4]

    json_data = {
        # parameters
        'neutral_temp_K': thruster_input['neutral_temp_K'],
        'neutral_velocity_m_s': thruster_input['u_n'],
        'ion_temp_K': thruster_input['ion_temp_K'],
        'cathode_electron_temp_eV': thruster_input['T_ec'],
        'sheath_loss_coefficient': thruster_input['c_w'],
        'inner_outer_transition_length_m': thruster_input['l_t'] * 1e-3,
        'anom_model_coeffs': anom_model_coeffs,
        'background_pressure_Torr': 10 ** thruster_input['PB'],
        'background_temperature_K': thruster_input['background_temperature_K'],
        'neutral_ingestion_multiplier': thruster_input['f_n'],
        'apply_thrust_divergence_correction': thruster_input['apply_thrust_divergence_correction'],
        # design
        'thruster_name': thruster_input['thruster_name'],
        'inner_radius': thruster_input['inner_radius'],
        'outer_radius': thruster_input['outer_radius'],
        'channel_length': thruster_input['channel_length'],
        'magnetic_field_file': str(CONFIG_DIR / thruster_input['magnetic_field_file']),
        'wall_material': thruster_input['wall_material'],
        'magnetically_shielded': thruster_input['magnetically_shielded'],
        'anode_potential': thruster_input['Va'],
        'cathode_potential': thruster_input['V_cc'],
        'anode_mass_flow_rate': thruster_input['mdot_a'] * 1e-6,
        'propellant': thruster_input['propellant_material'],
        # simulation                  
        'num_cells': thruster_input['num_cells'],
        'dt_s': thruster_input['dt_s'],
        'duration_s': thruster_input['duration_s'],
        'num_save': thruster_input['num_save'],
        'cathode_location_m': thruster_input['l_c'],
        'max_charge': thruster_input['max_charge'],
        'flux_function': thruster_input['flux_function'],
        'limiter': thruster_input['limiter'],
        'reconstruct': thruster_input['reconstruct'],
        'ion_wall_losses': thruster_input['ion_wall_losses'],
        'electron_ion_collisions': thruster_input['electron_ion_collisions'],
        'anom_model': thruster_input['anom_model'],
    }
    
    if anom_model == 'ShiftedTwoZone' or anom_model == 'ShiftedGaussianBohm':
        # Add extra parameters for anomalous transport models that depend on pressure
        json_data.update({'pressure_dz': thruster_input['delta_z'] * thruster_input['channel_length'],
                          'pressure_z0': thruster_input['z0'] * thruster_input['channel_length'],
                          'pressure_pstar': thruster_input['p0'] * 1e-6,
                          'pressure_alpha': thruster_input['alpha']})
    return json_data


def hallthruster_jl_model(thruster_input: dict, jl=None) -> dict:
    """Run a single Hallthruster.jl simulation for a given set of inputs.

    :param thruster_input: named key-value pairs of thruster inputs
    :param jl: an instance of `julicall.Main` for running Julia code from within Python
    :raises ModelRunException: if anything fails in `juliacall`
    :returns: `dict` of Hallthruster.jl outputs for this input
    """
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
        t1 = time.time()
        sol = jl.seval(f'sol = HallThruster.run_simulation("{fd.name}", verbose=False)')
        os.unlink(fd.name)   # delete the tempfile
    except juliacall.JuliaError as e:
        raise ModelRunException(f"Julicall error in Hallthruster.jl: {e}")

    if str(sol.retcode).lower() != "success":
        raise ModelRunException(f"Exception in Hallthruster.jl: Retcode = {sol.retcode}")

    # Average simulation results
    avg = jl.seval(f"avg = HallThruster.time_average(sol, {thruster_input['time_avg_frame_start']})")

    # Extract needed data
    I_B0 = jl.HallThruster.ion_current(avg)[0]
    niui_exit = 0.0
    ni_exit = 0.0
    for Z in range(avg.params.ncharge):
        ni_exit += jl.seval(f"avg[:ni, {Z}][][end]")
        niui_exit += jl.seval(f"avg[:niui, Z][][end]")
    end
    ui_avg = niui_exit / ni_exit
    
    # Load simulation results
    fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
    fd.close()
    
    jl.HallThruster.write_to_json(fd.name, avg)
    with open(fd.name, 'r') as f:
        thruster_output = json.load(f)
    os.unlink(fd.name)  # delete the tempfile

    thrust = thruster_output[0]['thrust']
    discharge_current = thruster_output[0]['discharge_current']

    thruster_output[0].update({'ui_avg': ui_avg / 1000.0, 'I_B0': I_B0, 'T': thrust, 'I_D': discharge_current,
                               'eta_c': thruster_output[0]['current_eff'], 'eta_m': thruster_output[0]['mass_eff'],
                               'eta_v': thruster_output[0]['voltage_eff']})

    # Raise an exception if thrust or beam current are negative (non-physical cases)
    if thrust < 0 or I_B0 < 0:
        raise ModelRunException(f'Exception due to non-physical case: thrust={thrust} N, beam current={I_B0} A')

    return thruster_output[0]


def hallthruster_jl_wrapper(x: np.ndarray, alpha: tuple = (2, 2), *, compress: bool = False,
                            output_dir: str | Path = None, n_jobs: int = -1,
                            config: str | Path = CONFIG_DIR / 'hallthruster_jl.json',
                            variables: str | Path = CONFIG_DIR / 'variables_v0.json',
                            svd_data: dict | str | Path = CONFIG_DIR / 'thruster_svd.pkl',
                            hf_override: tuple | bool = None, thruster: str = 'SPT-100'):
    """Wrapper function for Hallthruster.jl.

    !!! Note "Defining input variables"
        This function loads variable definitions from the path specified in `variables`. The variables are loaded in
        the form of `BaseRV` objects from the `amisc` package. You can directly edit this config file to change the
        definitions of the variables or add new variables, or you can specify a different file.

    !!! Info "Dimension reduction"
        If you specify `compress=True`, then the `svd_data` will be used to compress the ion velocity profile. The
        default is a file named `thruster_svd.pkl` in the `config` directory. The format of the `svd_file` is a Python
        `.pkl` save file with the fields `A` &rarr; $N\\times M$ SVD data matrix and `vtr` &rarr; $r\\times M$ the
        linear projection matrix from high dimension $M$ to low dimension $r$. See the theory page for more details.

    :param x: `(..., xdim)` the model inputs, ordering is specified as "inputs" in the `config` file
    :param alpha: `($\\alpha_1$, $\\alpha_2$) model fidelity indices = ($N_{cells}$, $N_{charge}$)
    :param compress: Whether to compress the ion velocity profile with SVD dimension reduction
    :param output_dir: path where to save Hallthruster.jl result .json files, none saved if not specified
    :param n_jobs: number of jobs to run in parallel, use all available cpus if -1
    :param config: path to .json config file to load static thruster simulation configs (.json)
    :param variables: path to .json file that specifies all input variables
    :param svd_data: path to a .pkl file that is used to compress the ion velocity profile, can also directly pass in
                     the `dict` data from the .pkl file
    :param hf_override: the fidelity indices to override `alpha`
    :param thruster: the name of the thruster to simulate (must be defined in `config`)
    :returns: `dict(y, files, cost)`, the model outputs `y=(..., ydim)`, list of output files, and avg model cpu time;
                                      order of outputs in `ydim` is specified as "outputs" in the `config` file
    """
    x = np.atleast_1d(x)
    # Check for a single-fidelity override of alpha
    if isinstance(hf_override, tuple) and len(hf_override) == 2:
        alpha = hf_override
    elif hf_override:
        alpha = (2, 2)

    # Set model fidelity quantities from alpha
    Ncells = 50 * (alpha[0] + 2)
    Ncharge = alpha[1] + 1
    # dt_map = [25e-9, 12.5e-9, 8.4e-9, 6.3e-9]
    dt_map = [12.5e-9, 8.4e-9, 6.3e-9]
    dt_s = dt_map[alpha[0]] if Ncharge <= 2 else dt_map[alpha[0]] / np.sqrt(3/2)

    # Constant inputs from config file (thruster geometry, propellant, wall material, simulation params, etc.)
    with open(Path(config), 'r') as fd:
        config_data = json.load(fd)
        default_inputs = load_variables(config_data['default_inputs'], Path(variables))
        base_input = {var.id: var.nominal for var in default_inputs}  # Set default values for variables.json RV inputs
        base_input.update(config_data[thruster])                      # Set all other simulation configs
        input_list = config_data['required_inputs']  # Needs to match xdim and correspond with str input ids to hallthruster.jl
        output_list = config_data['outputs']
    base_input.update({'num_cells': Ncells, 'dt_s': dt_s, 'max_charge': Ncharge})  # Update model fidelity params

    # Load svd params for dimension reduction of ion velocity profile
    if compress:
        if not isinstance(svd_data, dict):
            with open(svd_data, 'rb') as fd:
                svd_data = pickle.load(fd)
        vtr = svd_data['vtr']  # (r x M)
        A = svd_data['A']
        A_mu = np.mean(A, axis=0)
        A_std = np.std(A, axis=0)
        r, M = vtr.shape
        ydim = r + len(output_list) - 1
    else:
        M = Ncells + 2
        ydim = M + len(output_list) - 1

    # Save the inputs to file
    eval_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    if output_dir is not None:
        save_dict = {'alpha': alpha, 'x': x}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

    def run_batch(job_num, index_batches, y):
        """Run a batch of indices into the input matrix `x`."""
        from juliacall import Main as jl
        jl.seval('using HallThruster')
        thruster_input = copy.deepcopy(base_input)
        curr_batch = index_batches[job_num]
        files = []  # Return an ordered list of output filenames corresponding to input indices
        costs = []  # Time to evaluate hallthruster.jl for a single input

        for i, index in enumerate(curr_batch):
            x_curr = [float(x[index + (i,)]) for i in range(x.shape[-1])]   # (xdim,)
            thruster_input.update({input_list[i]: x_curr[i] for i in range(x.shape[-1])})

            # Run hallthruster.jl
            t1 = time.time()
            try:
                res = hallthruster_jl_model(thruster_input, jl=jl)
            except ModelRunException as e:
                logger = get_logger(__name__)
                logger.warning(f'Skipping index {index} due to caught exception: {e}')
                y[index + (slice(None),)] = np.nan
                if output_dir is not None:
                    save_dict = {'input': thruster_input, 'Exception': str(e), 'index': index}
                    fname = f'{eval_id}_{index}_exc.json'
                    files.append(fname)
                    costs.append(0)
                    data_write(save_dict, fname, output_dir)
                continue

            # Save QoIs
            curr_idx = 0
            for i, qoi_str in enumerate(output_list):
                if qoi_str == 'uion':
                    if compress:
                        # Interpolate ion velocity to the full reconstruction grid (of dim M)
                        n_cells = M - 2  # M = num of grid pts = Ncells + 2 (half-grid cells at ends of FE domain)
                        L = thruster_input['l_c']  # Cathode location is the end of axial z domain
                        dz = L / n_cells
                        zg = np.zeros(M)  # zg is the axial z grid points for the reconstructed field (of size M)
                        zg[0] = 0
                        zg[1] = dz / 2
                        zg[2:-1] = zg[1] + np.arange(1, n_cells) * dz
                        zg[-1] = L
                        z1 = np.atleast_1d(res['z'])
                        ui1 = np.atleast_1d(res['ui_1'])
                        uig = np.interp(zg, z1, ui1)  # Interpolated ui on reconstruction grid (M,)
                        uig_r = np.squeeze(vtr @ ((uig - A_mu) / A_std)[..., np.newaxis], axis=-1)
                        y[index + (slice(curr_idx, curr_idx + r),)] = uig_r  # Compress to dim (r,)
                        curr_idx += r
                    else:
                        # Otherwise, save entire ion velocity grid
                        y[index + (slice(curr_idx, curr_idx + M),)] = res['ui_1']
                        curr_idx += M
                else:
                    # Append scalar qois
                    y[index + (curr_idx,)] = res[qoi_str]
                    curr_idx += 1
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
    y_ret = np.empty(y.shape)
    y_ret[:] = y[:]
    del y
    os.unlink(y_fd.name)

    # Re-order the resulting list of file names and costs
    files, costs = [], []
    flat_idx = 0
    for input_index in np.ndindex(*x.shape[:-1]):
        # Iterate in same circular fashion as the inputs were passed to parallel
        batch_files, batch_costs = res[flat_idx % num_batches]
        if output_dir is not None:
            files.append(batch_files.pop(0))
        costs.append(batch_costs.pop(0))
        flat_idx += 1

    # Save model eval summary to file
    if output_dir is not None:
        save_dict = {'alpha': alpha, 'x': x, 'y': y_ret, 'is_compressed': compress, 'files': files, 'costs': costs}
        with open(Path(output_dir) / f'{eval_id}_eval.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)
    costs = np.atleast_1d(costs).astype(np.float64)
    costs[costs == 0] = np.nan
    avg_model_cpu_time = np.nanmean(costs)

    return {'y': y_ret, 'files': files, 'cost': avg_model_cpu_time}


def uion_reconstruct(xr: np.ndarray, z: np.ndarray = None, L: float | np.ndarray = 0.08,
                     svd_data: dict | str | Path = CONFIG_DIR / 'thruster_svd.pkl') -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct an ion velocity profile, interpolate to `z` if provided.

    !!! Warning
        The `svd_data` must be the **same** as was used with `hallthruster_jl_wrapper` when compressing the data, i.e.
        the same SVD data must be used to reconstruct here.

    :param xr: `(... r)` The reduced dimension output of `hallthruster_jl_wrapper` (just the ion velocity profile)
    :param z: `(Nz,)` The axial `z` grid points to interpolate to (in meters, between 0 and `L`)
    :param L: `(...,)` The full domain length of the reconstructed grid(s)
    :param svd_data: path to a `.pkl` file that is used to compress/reconstruct the ion velocity profile, can also pass
                     the `dict` of svd data directly in
    :returns: `z, uion_interp` - `(..., Nz or M)` The reconstructed (and potentially interpolated) ion velocity
              profile(s), corresponds to `z=(0, 0.08)` m with `M=202` points by default
    """
    if z is not None:
        z = z.astype(xr.dtype)
    L = np.atleast_1d(L)
    interp_normal = len(L.shape) == 1 and L.shape[0] == 1

    # Load SVD data from file
    if not isinstance(svd_data, dict):
        with open(svd_data, 'rb') as fd:
            svd_data = pickle.load(fd)
    vtr = svd_data['vtr']       # (r x M)
    A = svd_data['A']
    A_mu = np.mean(A, axis=0)
    A_std = np.std(A, axis=0)
    r, M = vtr.shape

    n_cells = M - 2
    dz = L / n_cells
    zg = np.zeros(L.shape + (M,))  # zg is the axial z grid points for the reconstructed field (of size M)
    zg[..., 1] = dz / 2
    zg[..., 2:-1] = zg[..., 1, np.newaxis] + np.arange(1, n_cells) * dz[..., np.newaxis]
    zg[..., -1] = L
    uion_g = (np.squeeze(vtr.T @ xr[..., np.newaxis], axis=-1) * A_std + A_mu).astype(xr.dtype)      # (..., M)
    zg = (np.squeeze(zg, axis=0) if interp_normal else zg).astype(xr.dtype)    # (..., M)

    # Do vectorized 1d linear interpolation
    if z is not None:
        diff = zg[..., np.newaxis] - z                          # (..., M, Nz)
        lower_idx = np.argmin(np.abs(diff), axis=-2)            # (..., Nz)
        diff = np.take_along_axis(zg, lower_idx, axis=-1) - z
        lower_idx[diff > 0] -= 1
        upper_idx = lower_idx + 1
        lower_idx[lower_idx < 0] = 0
        upper_idx[upper_idx >= zg.shape[-1]] = zg.shape[-1] - 1
        x_lower = np.take_along_axis(zg, lower_idx, axis=-1)
        x_upper = np.take_along_axis(zg, upper_idx, axis=-1)
        if interp_normal:
            y_lower = uion_g[..., lower_idx]
            y_upper = uion_g[..., upper_idx]
        else:
            # Vectorized 1d interpolation
            y_lower = np.take_along_axis(uion_g, lower_idx, axis=-1)
            y_upper = np.take_along_axis(uion_g, upper_idx, axis=-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            uion_interp = y_lower + (z - x_lower) * (y_upper-y_lower) / (x_upper-x_lower)       # (..., Nz)

        # Set points outside grid equal to outer values
        lower_idx = z < zg[..., 0, np.newaxis]      # (..., Nz)
        upper_idx = z > zg[..., -1, np.newaxis]     # (..., Nz)
        if interp_normal:
            if np.any(lower_idx):
                uion_interp[..., lower_idx] = uion_g[..., 0, np.newaxis]
            if np.any(upper_idx):
                uion_interp[..., upper_idx] = uion_g[..., -1, np.newaxis]
        else:
            uion_interp[lower_idx] = uion_g[np.any(lower_idx, axis=-1), 0]
            uion_interp[upper_idx] = uion_g[np.any(upper_idx, axis=-1), -1]

        return z, uion_interp
    else:
        return zg, uion_g
