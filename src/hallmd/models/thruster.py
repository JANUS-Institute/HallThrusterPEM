"""Module for Hall thruster models.

!!! Note
    Only current implementation is for the 1d fluid [Hallthruster.jl code](https://github.com/UM-PEPL/HallThruster.jl).
    Other thruster codes can be implemented similarly here.

Includes:

- `hallthruster_jl()` - Wrapper to run HallThruster.jl for a single set of inputs
- `PEM_TO_JULIA` - Mapping of PEM variable names to a path in the HallThruster.jl input/output structure (defaults)
"""
import subprocess
from importlib import resources
from pathlib import Path
import time
import copy
import json
import tempfile
import os
import random
import string
import warnings
from typing import Literal, Callable

import numpy as np
from amisc.typing import Dataset
from scipy.interpolate import interp1d

from hallmd.utils import load_device

__all__ = ['hallthruster_jl', 'PEM_TO_JULIA']


# Check for Julia and HallThruster.jl
try:
    subprocess.run(["julia", "-e", 'using HallThruster; println("Success!")'],
                   check=True, capture_output=True, text=True)
except FileNotFoundError:
    warnings.warn("Julia executable not found. See https://julialang.org/downloads/. You may run into issues"
                  " while using hallmd.thruster")
except subprocess.CalledProcessError as e:
    warnings.warn("HallThruster.jl not found. See https://um-pepl.github.io/HallThruster.jl/dev/. "
                  "You may run into issues while using hallmd.thruster")
except Exception as e:
    warnings.warn(f"Failed to find Julia and HallThruster.jl for unknown reason: {e}\n "
                  f"You may run into issues while using hallmd.thruster")

# Maps PEM variable names to a path in the HallThruster.jl input/output structure (default values here)
with open(resources.files('hallmd.models') / 'pem_to_julia.json', 'r') as fd:
    PEM_TO_JULIA = json.load(fd)


def _convert_to_julia(pem_data: dict, julia_data: dict, pem_to_julia: dict):
    """Replace all values in the mutable `julia_data` dict with corresponding values in `pem_data`, using the
    conversion map provided in `pem_to_julia`. Will "blaze" a path into `julia_data` dict if it does not exist.

    :param pem_data: thruster inputs of the form `{'pem_variable_name': value}`
    :param julia_data: a `HallThruster.jl` data structure of the form `{'config': {...}, 'simulation': {...}, etc.}`
    :param pem_to_julia: a conversion map from a pem variable name to a path in `julia_data`. Use strings for dict keys
                         and integers for list indices. For example, `{'P_b': ['config', 'background_pressure', 2]}`
                         will set `julia_data['config']['background_pressure'][2] = pem_data['P_b']`.
    """
    for pem_key, value in pem_data.items():
        if pem_key not in pem_to_julia:
            raise KeyError(f'Cannot convert PEM data variable {pem_key} since it is not in the provided conversion map')

        # Blaze a trail into the julia data structure (set dict defaults for str keys and list defaults for int keys)
        julia_path = pem_to_julia.get(pem_key)
        pointer = julia_data
        for i, key in enumerate(julia_path[:-1]):
            if isinstance(pointer, dict) and not pointer.get(key):
                pointer.setdefault(key, {} if isinstance(julia_path[i+1], str) else [])
            if isinstance(pointer, list) and len(pointer) <= key:
                pointer.extend([{} if isinstance(julia_path[i+1], str) else [] for _ in range(key - len(pointer) + 1)])
            pointer = pointer[key]
        pointer[julia_path[-1]] = value


def _convert_to_pem(julia_data: dict, pem_to_julia: dict):
    """Return a `dict` of PEM outputs from a `HallThruster.jl` output structure, using the conversion map provided."""
    pem_data = {}
    for pem_key, julia_path in pem_to_julia.items():
        pointer = julia_data
        if julia_path[0] == 'outputs':
            found_value = True
            for key in julia_path:
                try:
                    pointer = pointer[key]
                except (KeyError, IndexError):
                    found_value = False
                    break

            if found_value:
                pem_data[pem_key] = pointer
    return pem_data


def _default_model_fidelity(model_fidelity: tuple) -> dict:
    """Built-in (default) method to convert model fidelity tuple to `ncells` and `ncharge` via:

    ```python
    ncells = 50 * (model_fidelity[0] + 2)
    ncharge = model_fidelity[1] + 1
    ```

    Also adjusts the time step `dt` to maintain the CFL condition.
    """
    ncells = 50 * (model_fidelity[0] + 2)
    ncharge = model_fidelity[1] + 1
    dt_map = [12.5e-9, 8.4e-9, 6.3e-9]
    dt_s = dt_map[model_fidelity[0]] if ncharge <= 2 else dt_map[model_fidelity[0]] / np.sqrt(3 / 2)

    return {'ncells': ncells, 'ncharge': ncharge, 'dt': dt_s}


def _format_hallthruster_jl_input(thruster_inputs: dict, thruster: dict | str, config: dict, simulation: dict,
                                  postprocess: dict, model_fidelity: tuple, output_path: str | Path,
                                  pem_to_julia: dict, fidelity_function: Callable) -> dict:
    """Helper function to format inputs for `Hallthruster.jl` as a `dict` writeable to `json`. See the call
    signature of `hallthruster_jl` for more details on arguments.

    The return `dict` has the format:
    ```json
    {
        "config": {
            "thruster": {...},
            "discharge_voltage": 300,
            "domain": [0, 0.08],
            "anode_mass_flow_rate": 1e-6,
            "background_pressure": 1e-5,
            etc.
        },
        "simulation": {
            "ncells": 202,
            "dt": 8.4e-9
            "duration": 1e-3,
            etc.
        }
        "postprocess": {...}
    }
    ```

    :returns: a json `dict` in the format that `HallThruster.run_simulation()` expects to be called
    """
    json_config = {'config': {} if config is None else copy.deepcopy(config),
                   'simulation': {} if simulation is None else copy.deepcopy(simulation),
                   'postprocess': {} if postprocess is None else copy.deepcopy(postprocess)}

    # Necessary to load thruster specs separately from the config (to protect sensitive data)
    if isinstance(thruster, str):
        thruster = load_device(thruster)
    if thruster is not None:
        json_config['config']['thruster'] = thruster  # override

    # Override model fidelity quantities
    if model_fidelity is not None:
        _convert_to_julia(fidelity_function(model_fidelity), json_config, pem_to_julia)

    def _random_filename():
        fname = f'hallthruster_jl'
        if name := json_config['config'].get('thruster', {}).get('name'):
            fname += f'_{name}'
        if vd := json_config['config'].get('discharge_voltage'):
            fname += f'_{round(vd)}V'
        if mdot := json_config['config'].get('anode_mass_flow_rate'):
            fname += f'_{mdot:.1e}kg_s'

        fname += '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) + '.json'
        return fname

    # Make sure we request time-averaged and output file name
    duration = json_config['simulation'].get('duration', 1e-3)
    avg_start_time = json_config['postprocess'].get('average_start_time', 0.5 * duration)
    output_file = _random_filename()
    if output_path is not None:
        output_file = str((Path(output_path) / output_file).resolve())

    json_config['postprocess']['average_start_time'] = avg_start_time
    json_config['postprocess']['output_file'] = output_file

    # Update/override config with PEM thruster inputs (modify in place)
    _convert_to_julia(thruster_inputs, json_config, pem_to_julia)

    # Handle special conversions for anomalous transport models (just c1*c2 Bohm model for now)
    if anom_model := json_config['config'].get('anom_model'):
        if anom_model.get('type') == 'PressureShifted':
            anom_model = anom_model.get('model', {})

        match anom_model.get('type', 'TwoZoneBohm'):
            case 'TwoZoneBohm':
                anom_model['c1'] = anom_model.get('c1', 0.00625)
                anom_model['c2'] = anom_model['c2'] * anom_model['c1'] if thruster_inputs.get('a_2') is not None \
                    else anom_model.get('c2', 0.0625)
            case 'GaussianBohm':
                pass

    # Make sure we have all required simulation configs
    for required_config in ['thruster', 'discharge_voltage', 'domain', 'anode_mass_flow_rate']:
        if required_config not in json_config['config']:
            raise ValueError(f"Missing required simulation config: {required_config}")
    for required_config in ['duration', 'ncells', 'dt']:
        if required_config not in json_config['simulation']:
            raise ValueError(f"Missing required simulation config: {required_config}")

    return json_config


def hallthruster_jl(thruster_inputs: Dataset,
                    thruster: Literal['SPT-100'] | str | dict = 'SPT-100',
                    config: dict = None,
                    simulation: dict = None,
                    postprocess: dict = None,
                    u_ion_coords: np.ndarray = None,
                    model_fidelity: tuple = (2, 2),
                    output_path: str | Path = None,
                    julia_script: str | Path = 'default',
                    pem_to_julia: dict = 'default',
                    fidelity_function: Callable[[tuple[int, ...]], dict] = 'default') -> Dataset:
    """Run a single `HallThruster.jl` simulation for a given set of inputs. This function will write a temporary
    input file to disk, call `HallThruster.run_simulation()` in Julia, and read the output file back into Python. Will
    return time-averaged performance metrics and ion velocity for use with the PEM.

    !!! Warning "Required configuration"
        You must specify a thruster, a domain, a mass flow rate, and a discharge voltage to run the simulation. The
        thruster must be defined in the `hallmd.devices` directory or as a dictionary with the required fields.
        The mass flow rate and discharge voltage are specified in `thruster_inputs` as `mdot_a` (kg/s) and
        `V_a` (V), respectively. The domain is specified as a list `[left_bound, right_bound]` in the
        `config` dictionary. See the
        [HallThruster.jl docs](https://um-pepl.github.io/HallThruster.jl/stable/config/) for more details.

    :param thruster_inputs: named key-value pairs of thruster inputs: `P_b`, `V_a`, `mdot_a`, `T_e`, `u_n`, `l_t`,
                            `a_1`, `a_2`, `delta_z`, `z0`, `p0`, and `V_cc` for background pressure (Torr), anode
                            voltage, anode mass flow rate (kg/s), electron temperature (eV), neutral velocity (m/s),
                            transition length (m), anomalous transport coefficients, and cathode coupling voltage. Will
                            override the corresponding values in `config` if provided.
    :param thruster: the name of the thruster to simulate (must be importable from `hallmd.devices`, see
                     [`load_device`][hallmd.utils.load_device]), or a dictionary that provides geometry and
                     magnetic field information of the thruster to simulate; see the
                     [Hallthruster.jl docs](https://um-pepl.github.io/HallThruster.jl/dev/run/). Will override
                     `thruster` in `config` if provided. If None, will defer to `config`. Defaults to the SPT-100.
    :param config: dictionary of configs for `HallThruster.jl`, see the
                   [Hallthruster.jl docs](https://um-pepl.github.io/HallThruster.jl/stable/config/) for
                   options and formatting.
    :param simulation: dictionary of simulation parameters for `HallThruster.jl`
    :param postprocess: dictionary of post-processing parameters for `Hallthruster.jl`
    :param u_ion_coords: `(M,)` The axial locations at which to compute the ion velocity profile, in meters. Defaults
                         to the full `ncells` simulation grid computed by `HallThruster.jl`. `z=0` corresponds to the
                         anode at the left BC of the simulation domain. If provided, must be within the `domain` bounds
                         specified in `config`.
    :param model_fidelity: tuple of integers that determine the number of cells and the number of charge states to use
                           via `ncells = model_fidelity[0] * 50 + 100` and `ncharge = model_fidelity[1] + 1`.
                           Will override `ncells` and `ncharge` in `simulation` and `config` if provided.
    :param output_path: base path to save output files, will write to current directory if not specified
    :param julia_script: path to a custom Julia script to run instead of the default `run_hallthruster.jl` script
    :param pem_to_julia: a `dict` mapping of PEM shorthand variable names to a list of keys that maps into the
                         `HallThruster.jl` input/output data structure. Defaults to the provided PEM_TO_JULIA dict
                         defined in [`hallmd.models.thruster`][hallmd.models.thruster]. For example,
                         `{'P_b': ['config', 'background_pressure']}` will set `config['background_pressure'] = P_b`.
                         If specified, will override and extend the default mapping.
    :param fidelity_function: a callable that takes a tuple of integers and returns a dictionary of simulation
                              parameters. Defaults to `_convert_model_fidelity` which sets `ncells` and `ncharge` based
                              on the input tuple. The returned simulation parameters must be convertable to Julia via
                              the `pem_to_julia` mapping.
    :raises ModelRunException: if anything fails during the call to `Hallthruster.jl`
    :returns: `dict` of `Hallthruster.jl` outputs: `I_B0`, `I_d`, `T`, `eta_c`, `eta_m`, `eta_v`, and `u_ion` for ion
              beam current (A), discharge current (A), thrust (N), current efficiency, mass efficiency, voltage
              efficiency, and singly-charged ion velocity profile (m/s), all time-averaged
    """
    if julia_script is None or julia_script == 'default':
        julia_script = resources.files('hallmd.models') / 'run_hallthruster.jl'
    if fidelity_function is None or fidelity_function == 'default':
        fidelity_function = _default_model_fidelity
    if output_path is None:
        output_path = Path.cwd()
    if pem_to_julia is None or pem_to_julia == 'default':
        pem_to_julia = copy.deepcopy(PEM_TO_JULIA)
    else:
        tmp = copy.deepcopy(PEM_TO_JULIA)
        tmp.extend(pem_to_julia)
        pem_to_julia = tmp

    # Format inputs for HallThruster.jl and write to json
    json_data = _format_hallthruster_jl_input(thruster_inputs, thruster, config, simulation, postprocess,
                                              model_fidelity, output_path, pem_to_julia, fidelity_function)
    fd = tempfile.NamedTemporaryFile(suffix='.json', encoding='utf-8', mode='w', delete=False)
    json.dump(json_data, fd, ensure_ascii=False, indent=4)
    fd.close()

    # Run simulation
    try:
        t1 = time.time()
        subprocess.run(['julia', str(Path(julia_script).resolve()), fd.name], check=True)
        t2 = time.time()
    finally:
        os.unlink(fd.name)   # delete the tempfile

    # Load simulation results
    output_file = json_data['postprocess'].get('output_file')
    with open(Path(output_file), 'r') as fd:
        sim_results = json.load(fd)

    # Format QoIs for the PEM
    thruster_outputs = _convert_to_pem(sim_results, pem_to_julia)

    # Interpolate ion velocity to requested coords
    if u_ion_coords is not None:
        z_grid = thruster_outputs.get('u_ion_coords')
        uion_grid = thruster_outputs.get('u_ion')
        if z_grid is not None and uion_grid is not None:
            f = interp1d(np.atleast_1d(z_grid), np.atleast_1d(uion_grid), axis=-1)
            thruster_outputs['u_ion'] = f(u_ion_coords)  # (..., num_pts)
            thruster_outputs['u_ion_coords'] = u_ion_coords

    # Raise an exception if thrust or beam current are negative (non-physical cases)
    thrust = thruster_outputs.get('T', 0)
    beam_current = thruster_outputs.get('I_B0', 0)
    if thrust < 0 or beam_current < 0:
        raise ValueError(f'Exception due to non-physical case: thrust={thrust} N, '
                         f'beam current={beam_current} A')

    thruster_outputs['model_cost'] = t2 - t1  # seconds
    thruster_outputs['output_path'] = Path(output_file).relative_to(Path(output_path).resolve()).as_posix()

    return thruster_outputs
