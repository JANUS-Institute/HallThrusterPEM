"""Module for Hall thruster models.

!!! Note
    Only current implementation is for the 1d fluid [Hallthruster.jl code](https://github.com/UM-PEPL/HallThruster.jl).
    Other thruster codes can be implemented similarly here.

Includes:

- `hallthruster_jl_model()` - Wrapper to run HallThruster.jl for a single set of inputs
- `INPUTS_TO_JL` - Mapping of PEM input names to HallThruster.jl input names
- `OUTPUTS_TO_JL` - Mapping of PEM output names to HallThruster.jl output names
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
from typing import Literal

import numpy as np
from amisc.typing import Dataset
from scipy.interpolate import interp1d

from hallmd.utils import load_device

__all__ = ['hallthruster_jl_model']


INPUTS_TO_JL = {
    'P_b': 'background_pressure',       # Torr
    'mdot_a': 'anode_mass_flow_rate',   # kg/s
    'V_cc': 'cathode_potential',        # V
    'u_n': 'neutral_velocity',          # m/s
    'T_e': 'cathode_Te',                # eV
    'l_t': 'transition_length',         # m
    'V_a': 'discharge_voltage',         # V
    # anom coefficients are handled separately
}

OUTPUTS_TO_JL = {
    'I_B0': 'ion_current',       # A
    'I_d': 'discharge_current',  # A
    'T': 'thrust',               # N
    'eta_c': 'current_eff',      # -
    'eta_m': 'mass_eff',         # -
    'eta_v': 'voltage_eff',      # -
    'eta_a': 'anode_eff',        # -
    'u_ion': 'ui_1',             # m/s
}


def _format_hallthruster_jl_input(thruster_inputs: dict, thruster: dict | str, config: dict, simulation: dict,
                                  postprocess: dict, model_fidelity: tuple, output_path: str | Path) -> dict:
    """Helper function to format inputs for `Hallthruster.jl` as a `dict` writeable to `json`. See the call
    signature of `hallthruster_jl_model` for more details on arguments.

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

    # Necessary to load thruster specs separately from the simulation config (to protect sensitive data)
    if isinstance(thruster, str):
        thruster = load_device(thruster)
    if thruster is not None:
        json_config['config']['thruster'] = thruster  # override

    # Override model fidelity quantities
    if model_fidelity is not None:
        ncells = 50 * (model_fidelity[0] + 2)
        ncharge = model_fidelity[1] + 1
        dt_map = [12.5e-9, 8.4e-9, 6.3e-9]
        dt_s = dt_map[model_fidelity[0]] if ncharge <= 2 else dt_map[model_fidelity[0]] / np.sqrt(3 / 2)
        json_config['simulation']['ncells'] = ncells
        json_config['simulation']['dt'] = dt_s
        json_config['config']['ncharge'] = ncharge

    # Update/override config with PEM thruster inputs
    for pem_key, jl_key in INPUTS_TO_JL.items():
        if pem_val := thruster_inputs.get(pem_key):
            json_config['config'][jl_key] = pem_val

    # Handle anomalous transport model
    if anom_model := json_config['config'].get('anom_model'):
        match anom_model.get('type', 'TwoZoneBohm'):
            case 'TwoZoneBohm':
                c1, c2 = anom_model.get('params', (0.00625, 0.0625))
                c1 = thruster_inputs.get('a_1', c1)
                c2 = thruster_inputs.get('a_2', c2)
                anom_model['params'] = [c1, c2]
            case 'ShiftedTwoZoneBohm':
                c1, c2, z0, dz, alpha, pstar = anom_model.get('params', (0.00625, 0.0625, -0.12, 0.2, 15, 45e-6))
                c1 = thruster_inputs.get('a_1', c1)
                c2 = thruster_inputs.get('a_2', c2)
                z0 = thruster_inputs.get('z0', z0)
                dz = thruster_inputs.get('delta_z', dz)
                alpha = thruster_inputs.get('alpha', alpha)
                pstar = thruster_inputs.get('p0', pstar)
                anom_model['params'] = [c1, c2, z0, dz, alpha, pstar]
            case _:
                raise ValueError(f"Unknown anomalous transport model: {anom_model['type']}")

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

    # Make sure we request performance metrics, time-averaged, and output file name
    duration = json_config['simulation'].get('duration', 1e-3)
    avg_start_time = json_config['postprocess'].get('average_start_time', 0.5 * duration)
    metrics = json_config['postprocess'].get('metrics', [])
    for required_metric in ['thrust', 'discharge_current', 'ion_current', 'current_eff', 'mass_eff', 'voltage_eff',
                            'anode_eff']:
        if required_metric not in metrics:
            metrics.append(required_metric)
    json_config['postprocess']['metrics'] = metrics
    json_config['postprocess']['average_start_time'] = avg_start_time

    output_file = _random_filename()
    if output_path is not None:
        output_file = str((Path(output_path) / output_file).resolve())
    json_config['postprocess']['output_file'] = output_file

    # Make sure we have all required simulation configs
    for required_config in ['thruster', 'discharge_voltage', 'domain', 'anode_mass_flow_rate']:
        if required_config not in json_config['config']:
            raise ValueError(f"Missing required simulation config: {required_config}")
    for required_config in ['duration', 'ncells', 'dt']:
        if required_config not in json_config['simulation']:
            raise ValueError(f"Missing required simulation config: {required_config}")

    return json_config


def hallthruster_jl_model(thruster_inputs: Dataset,
                          thruster: Literal['SPT-100'] | str | dict = 'SPT-100',
                          config: dict = None,
                          simulation: dict = None,
                          postprocess: dict = None,
                          julia_script: str | Path = None,
                          u_ion_coords: np.ndarray = None,
                          model_fidelity: tuple = (2, 2),
                          output_path: str | Path = None) -> Dataset:
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
                            override the corresponding values in `simulation_config` if provided.
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
    :param julia_script: path to a custom Julia script to run instead of the default `run_hallthruster.jl` script
    :param u_ion_coords: `(M,)` The axial locations at which to compute the ion velocity profile, in meters. Defaults
                         to the full `ncells` simulation grid computed by `HallThruster.jl`. `z=0` corresponds to the
                         anode at the left BC of the simulation domain. If provided, must be within the `domain` bounds
                         specified in `config`.
    :param model_fidelity: tuple of integers that determine the number of cells and the number of charge states to use
                           via `ncells = model_fidelity[0] * 50 + 100` and `ncharge = model_fidelity[1] + 1`.
                           Will override `ncells` and `ncharge` in `simulation` and `config` if provided.
    :param output_path: base path to save output files, will write to current directory if not specified
    :raises ModelRunException: if anything fails during the call to `Hallthruster.jl`
    :returns: `dict` of `Hallthruster.jl` outputs: `I_B0`, `I_d`, `T`, `eta_c`, `eta_m`, `eta_v`, and `u_ion` for ion
              beam current (A), discharge current (A), thrust (N), current efficiency, mass efficiency, voltage
              efficiency, and singly-charged ion velocity profile (m/s), all time-averaged
    """
    # Check for Julia and HallThruster.jl (will fail when trying to run the simulation anyways)
    # if not shutil.which("julia"):
    #     raise RuntimeError("Julia binary not found on system path. Please install Julia and ensure "
    #                        "it is on the system path.")
    # try:
    #     subprocess.run(["julia", "-e", "using HallThruster"], check=True, capture_output=True)
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError(f"HallThruster.jl library not found or failed to load: {e.stderr.decode()}") from e

    if julia_script is None:
        julia_script = resources.files('hallmd.models') / 'run_hallthruster.jl'
    if output_path is None:
        output_path = Path.cwd()

    # Format inputs for HallThruster.jl and write to json
    json_data = _format_hallthruster_jl_input(thruster_inputs, thruster, config, simulation, postprocess,
                                              model_fidelity, output_path)
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
    thruster_outputs = {pem_key: sim_results['outputs']['average'][jl_key] for pem_key, jl_key in OUTPUTS_TO_JL.items()}

    # Interpolate ion velocity to requested coords
    if u_ion_coords is not None:
        z_grid = np.atleast_1d(sim_results['outputs']['average']['z'])
        uion_grid = np.atleast_1d(thruster_outputs['u_ion'])
        f = interp1d(z_grid, uion_grid, axis=-1)
        thruster_outputs['u_ion'] = f(u_ion_coords)  # (..., num_pts)

    # Raise an exception if thrust or beam current are negative (non-physical cases)
    if thruster_outputs['T'] < 0 or thruster_outputs['I_B0'] < 0:
        raise ValueError(f'Exception due to non-physical case: thrust={thruster_outputs["T"]} N, '
                         f'beam current={thruster_outputs["I_B0"]} A')

    thruster_outputs['model_cost'] = t2 - t1  # seconds
    thruster_outputs['output_path'] = Path(output_file).relative_to(Path(output_path)).as_posix()

    return thruster_outputs
