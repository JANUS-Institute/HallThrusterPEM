"""Module for Hall thruster models.

!!! Note
    Only current implementation is for the 1d fluid [Hallthruster.jl code](https://github.com/UM-PEPL/HallThruster.jl).
    Other thruster codes can be implemented similarly here.

Includes:

- `run_hallthruster_jl` - General wrapper to run HallThruster.jl for a single set of inputs
- `hallthruster_jl()` - PEM wrapper to run HallThruster.jl for a set of PEM inputs
- `get_jl_env` - Get the path of the julia environment created for HallThruster.jl for a specific git ref
- `PEM_TO_JULIA` - Mapping of PEM variable names to a path in the HallThruster.jl input/output structure (defaults)
"""
import copy
import json
import os
import random
import string
import subprocess
import tempfile
import time
import warnings
from importlib import resources
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from amisc.typing import Dataset

from hallmd.utils import AVOGADRO_CONSTANT, FUNDAMENTAL_CHARGE, MOLECULAR_WEIGHTS, load_device

__all__ = ['run_hallthruster_jl', 'hallthruster_jl', 'get_jl_env', 'PEM_TO_JULIA']

HALLTHRUSTER_VERSION_DEFAULT = "0.18.1"

# Maps PEM variable names to a path in the HallThruster.jl input/output structure (default values here)
with open(resources.files('hallmd.models') / 'pem_to_julia.json', 'r') as fd:
    PEM_TO_JULIA = json.load(fd)


def get_jl_env(git_ref: str) -> Path:
    """Get the path of the julia environment created for HallThruster.jl for a specific git ref.

    :param git_ref: The git ref (i.e. commit hash, version tag, branch, etc.) of HallThruster.jl to use.
    """
    global_env_dir = Path('~/.julia/environments/').expanduser()
    env_path = global_env_dir / f"hallthruster_{git_ref}"
    return env_path


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
        if julia_path[0] == 'output':
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


def _default_model_fidelity(model_fidelity: tuple, json_config: dict, cfl: float = 0.2) -> dict:
    """Built-in (default) method to convert model fidelity tuple to `ncells` and `ncharge` via:

    ```python
    ncells = 50 * (model_fidelity[0] + 2)
    ncharge = model_fidelity[1] + 1
    ```

    Also adjusts the time step `dt` to maintain the CFL condition (based on grid spacing and ion velocity).

    :param model_fidelity: tuple of integers that determine the number of cells and the number of charge states to use
    :param json_config: the current set of configurations for HallThruster.jl
    :param cfl: a conservative estimate for CFL condition to determine time step
    :returns: a dictionary of simulation parameters that can be converted to Julia via the `pem_to_julia` mapping,
              namely `{'num_cells': int, 'ncharge': int, 'dt': float}`
    """
    if model_fidelity == ():
        model_fidelity = (2, 2)  # default to high-fidelity model

    num_cells = 50 * (model_fidelity[0] + 2)
    ncharge = model_fidelity[1] + 1

    # Estimate conservative time step based on CFL and grid spacing
    config = json_config.get('config', {})
    domain = config.get('domain', [0, 0.08])
    anode_pot = config.get('discharge_voltage', 300)
    cathode_pot = config.get('cathode_coupling_voltage', 0)
    propellant = config.get('propellant', 'Xenon')

    if propellant not in MOLECULAR_WEIGHTS:
        warnings.warn(f"Could not find propellant {propellant} in `hallmd.utils`. "
                      f"Will default to Xenon for estimating uniform time step from CFL...")
        propellant = 'Xenon'

    mi = MOLECULAR_WEIGHTS[propellant] / AVOGADRO_CONSTANT / 1000  # kg
    dx = float(domain[1]) / (num_cells + 1)
    u = np.sqrt(2 * ncharge * FUNDAMENTAL_CHARGE * (anode_pot - cathode_pot) / mi)
    dt_s = cfl * dx / u

    return {'num_cells': num_cells, 'ncharge': ncharge, 'dt': float(dt_s)}


def _format_hallthruster_jl_input(thruster_inputs: dict,
                                  thruster: dict | str = 'SPT-100',
                                  config: dict = None,
                                  simulation: dict = None,
                                  postprocess: dict = None,
                                  model_fidelity: tuple = (2, 2),
                                  output_path: str | Path = None,
                                  pem_to_julia: dict = 'default',
                                  fidelity_function: Callable = 'default') -> dict:
    """Helper function to format PEM inputs for `Hallthruster.jl` as a `dict` writeable to `json`. See the call
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
    if fidelity_function is None or fidelity_function == 'default':
        fidelity_function = _default_model_fidelity

    json_config = {'config': {} if config is None else copy.deepcopy(config),
                   'simulation': {} if simulation is None else copy.deepcopy(simulation),
                   'postprocess': {} if postprocess is None else copy.deepcopy(postprocess)}

    # Necessary to load thruster specs separately from the config (to protect sensitive data)
    if isinstance(thruster, str):
        thruster = load_device(thruster)
    if thruster is not None:
        json_config['config']['thruster'] = thruster  # override

    # Make sure we request time-averaged
    duration = json_config['simulation'].get('duration', 1e-3)
    avg_start_time = json_config['postprocess'].get('average_start_time', 0.5 * duration)
    json_config['postprocess']['average_start_time'] = avg_start_time

    # Update/override config with PEM thruster inputs (modify in place)
    _convert_to_julia(thruster_inputs, json_config, pem_to_julia)

    # Override model fidelity quantities
    if model_fidelity is not None:
        fidelity_overrides = fidelity_function(model_fidelity, json_config)
        _convert_to_julia(fidelity_overrides, json_config, pem_to_julia)

    if output_path is not None:
        fname = 'hallthruster_jl'
        if name := json_config['config'].get('thruster', {}).get('name'):
            fname += f'_{name}'
        if vd := json_config['config'].get('discharge_voltage'):
            fname += f'_{round(vd)}V'
        if mdot := json_config['config'].get('anode_mass_flow_rate'):
            fname += f'_{mdot:.1e}kg_s'

        fname += '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) + '.json'
        output_file = str((Path(output_path) / fname).resolve())
        json_config['postprocess']['output_file'] = output_file

    # Handle special conversions for anomalous transport models (just c1*c2 Bohm model for now)
    if anom_model := json_config['config'].get('anom_model'):
        if anom_model.get('type') == 'LogisticPressureShift':
            anom_model = anom_model.get('model', {})

        match anom_model.get('type', 'TwoZoneBohm'):
            case 'TwoZoneBohm':
                anom_model['c1'] = anom_model.get('c1', 0.00625)
                anom_model['c2'] = anom_model['c2'] * anom_model['c1'] if thruster_inputs.get('a_2') is not None \
                    else anom_model.get('c2', 0.0625)
            case 'GaussianBohm':
                pass

    return json_config


def run_hallthruster_jl(json_input: dict | str | Path, jl_env: str | Path = None, jl_script: str | Path = None,
                        **kwargs) -> dict:
    """Python wrapper for `HallThruster.run_simulation(json_input)` in Julia.

    :param json_input: either a dictionary containing `config`, `simulation`, and `postprocess` options for
            HallThruster.jl, or a string/Path containing a path to a JSON file with those inputs.
    :param jl_env: The julia environment containing HallThruster.jl. Defaults to global Julia environment.
    :param jl_script: path to a custom Julia script to run. The script should accept the input json file path as
                      a command line argument. Defaults to just calling `HallThruster.run_simulation(input_file)`.
    :param kwargs: additional keyword arguments to pass to `subprocess.run` when calling the Julia script.

    :returns: `dict` of `Hallthruster.jl` outputs. The specific outputs depend on the settings
              provided in the `postprocess` dict in the input. If `postprocess['output_file']` is present,
              this function will also write the requested outputs and restart information to that file.
    """
    # Read JSON input from file if path provided
    if isinstance(json_input, str | Path):
        with open(json_input, 'r') as fp:
            json_input = json.load(fp)

    tempfile_args = dict(suffix=".json", prefix="hallthruster_jl_", mode="w", delete=False, encoding="utf-8")

    # Get output file path. If one not provided, create a temporary
    temp_out = False
    if 'output_file' in json_input.get('postprocess', {}):
        output_file = Path(json_input['postprocess'].get('output_file'))
    elif 'output_file' in json_input.get('input', {}).get('postprocess', {}):
        output_file = Path(json_input['input']['postprocess'].get('output_file'))
    else:
        temp_out = True
        fd_out = tempfile.NamedTemporaryFile(**tempfile_args)
        output_file = Path(fd_out.name)
        fd_out.close()

        if json_input.get('input'):
            json_input['input'].setdefault('postprocess', {})
            json_input['input']['postprocess']['output_file'] = str(output_file.resolve())
        else:
            json_input.setdefault('postprocess', {})
            json_input['postprocess']['output_file'] = str(output_file.resolve())

    # Dump input to temporary file
    fd = tempfile.NamedTemporaryFile(**tempfile_args)
    input_file = fd.name
    json.dump(json_input, fd, ensure_ascii=False, indent=4)
    fd.close()

    # Run HallThruster.jl on input file
    if jl_script is None:
        cmd = ['julia', '--startup-file=no', '-e',
               f'using HallThruster; HallThruster.run_simulation(raw"{input_file}")']
    else:
        cmd = ['julia', '--startup-file=no', '--',
               str(Path(jl_script).resolve()), input_file]

    if jl_env is not None:
        if Path(jl_env).exists():
            cmd.insert(1, f'--project={Path(jl_env).resolve()}')
        else:
            raise ValueError(f"Could not find Julia environment {jl_env}. Please create it first. "
                             f"See https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/scripts/install_hallthruster.py")

    try:
        subprocess.run(cmd, **kwargs)
    finally:
        # Delete temporary input file
        os.unlink(input_file)

    # Load output data
    with open(output_file, 'r') as fp:
        output_data = json.load(fp)

    if temp_out:
        os.unlink(output_file)
        if d := output_data.get('postprocess'):
            if 'output_file' in d:
                del d['output_file']
        if d := output_data.get('input'):
            if d2 := d.get('postprocess'):
                if 'output_file' in d2:
                    del d2['output_file']

    return output_data


def hallthruster_jl(thruster_inputs: Dataset = None,
                    thruster: Literal['SPT-100'] | str | dict = 'SPT-100',
                    config: dict = None,
                    simulation: dict = None,
                    postprocess: dict = None,
                    model_fidelity: tuple = (2, 2),
                    output_path: str | Path = None,
                    version: str = HALLTHRUSTER_VERSION_DEFAULT,
                    pem_to_julia: dict = 'default',
                    fidelity_function: Callable[[tuple[int, ...]], dict] = 'default',
                    julia_script: str | Path = None,
                    run_kwargs: dict = 'default') -> Dataset:
    """Run a single `HallThruster.jl` simulation for a given set of inputs. This function will write a temporary
    input file to disk, call `HallThruster.run_simulation()` in Julia, and read the output file back into Python. Will
    return time-averaged performance metrics and ion velocity for use with the PEM.

    Note that the specific inputs and outputs described here can be configured using the `pem_to_julia` dict.

    !!! Warning "Required configuration"
        You must specify a thruster, a domain, a mass flow rate, and a discharge voltage to run the simulation. The
        thruster must be defined in the `hallmd.devices` directory or as a dictionary with the required fields.
        The mass flow rate and discharge voltage are specified in `thruster_inputs` as `mdot_a` (kg/s) and
        `V_a` (V), respectively. The domain is specified as a list `[left_bound, right_bound]` in the
        `config` dictionary. See the
        [HallThruster.jl docs](https://um-pepl.github.io/HallThruster.jl/dev/reference/config/) for more details.

    :param thruster_inputs: named key-value pairs of thruster inputs: `P_b`, `V_a`, `mdot_a`, `T_e`, `u_n`, `l_t`,
                            `a_1`, `a_2`, `delta_z`, `z0`, `p0`, and `V_cc` for background pressure (Torr), anode
                            voltage, anode mass flow rate (kg/s), electron temperature (eV), neutral velocity (m/s),
                            transition length (m), anomalous transport coefficients, and cathode coupling voltage. Will
                            override the corresponding values in `config` if provided.
    :param thruster: the name of the thruster to simulate (must be importable from `hallmd.devices`, see
                     [`load_device`][hallmd.utils.load_device]), or a dictionary that provides geometry and
                     magnetic field information of the thruster to simulate; see the
                     [Hallthruster.jl docs](https://um-pepl.github.io/HallThruster.jl/dev/tutorials/simulation//run/).
                     Will override `thruster` in `config` if provided. If None, will defer to `config`.
                     Defaults to the SPT-100.
    :param config: dictionary of configs for `HallThruster.jl`, see the
                   [Hallthruster.jl docs](https://um-pepl.github.io/HallThruster.jl/dev/reference/config/) for
                   options and formatting.
    :param simulation: dictionary of simulation parameters for `HallThruster.jl`
    :param postprocess: dictionary of post-processing parameters for `Hallthruster.jl`
    :param model_fidelity: tuple of integers that determine the number of cells and the number of charge states to use
                           via `ncells = model_fidelity[0] * 50 + 100` and `ncharge = model_fidelity[1] + 1`.
                           Will override `ncells` and `ncharge` in `simulation` and `config` if provided.
    :param output_path: base path to save output files, will write to current directory if not specified
    :param version: version of HallThruster.jl to use (defaults to 0.18.1); will
                    search for a global `hallthruster_{version}` environment in the `~/.julia/environments/` directory.
                    Can also specify a specific git ref (i.e. branch, commit hash, etc.) to use from GitHub. If the
                    `hallthruster_{version}` environment does not exist, an error will be raised -- you should create
                    this environment first before using it.
    :param pem_to_julia: a `dict` mapping of PEM shorthand variable names to a list of keys that maps into the
                         `HallThruster.jl` input/output data structure. Defaults to the provided PEM_TO_JULIA dict
                         defined in [`hallmd.models.thruster`][hallmd.models.thruster]. For example,
                         `{'P_b': ['config', 'background_pressure']}` will set `config['background_pressure'] = P_b`.
                         If specified, will override and extend the default mapping.
    :param fidelity_function: a callable that takes a tuple of integers and returns a dictionary of simulation
                              parameters. Defaults to `_default_model_fidelity` which sets `ncells` and `ncharge` based
                              on the input tuple. The returned simulation parameters must be convertable to Julia via
                              the `pem_to_julia` mapping. The callable should also take in the current json config dict.
    :param julia_script: path to a custom Julia script to run. The script should accept the input json file path as
                         a command line argument. Defaults to just calling `HallThruster.run_simulation(input_file)`.
    :param run_kwargs: additional keyword arguments to pass to `subprocess.run` when calling the Julia script.
                       Defaults to `check=True`.
    :returns: `dict` of `Hallthruster.jl` outputs: `I_B0`, `I_d`, `T`, `eta_c`, `eta_m`, `eta_v`, and `u_ion` for ion
              beam current (A), discharge current (A), thrust (N), current efficiency, mass efficiency, voltage
              efficiency, and singly-charged ion velocity profile (m/s), all time-averaged.
    """
    if pem_to_julia is None or pem_to_julia == 'default':
        pem_to_julia = copy.deepcopy(PEM_TO_JULIA)
    else:
        tmp = copy.deepcopy(PEM_TO_JULIA)
        tmp.update(pem_to_julia)
        pem_to_julia = tmp

    thruster_inputs = thruster_inputs or {}

    # Format PEM inputs for HallThruster.jl
    json_data = _format_hallthruster_jl_input(thruster_inputs, thruster=thruster, config=config, simulation=simulation,
                                              postprocess=postprocess, model_fidelity=model_fidelity,
                                              output_path=output_path, pem_to_julia=pem_to_julia,
                                              fidelity_function=fidelity_function)
    # Get julia environment
    jl_environment = get_jl_env(version) if version is not None else None

    if run_kwargs is None:
        run_kwargs = {}
    elif run_kwargs == 'default':
        run_kwargs = {'check': True}

    # Run Julia
    t1 = time.time()
    sim_results = run_hallthruster_jl(json_data, jl_env=jl_environment, jl_script=julia_script, **run_kwargs)
    t2 = time.time()

    # Format QOIs for PEM
    thruster_outputs = _convert_to_pem(sim_results, pem_to_julia)

    # Raise an exception if thrust or beam current are negative (non-physical cases)
    thrust = thruster_outputs.get('T', 0)
    beam_current = thruster_outputs.get('I_B0', 0)
    if thrust < 0 or beam_current < 0:
        raise ValueError(f'Exception due to non-physical case: thrust={thrust} N, '
                         f'beam current={beam_current} A')

    thruster_outputs['model_cost'] = t2 - t1  # seconds

    if output_path is not None:
        output_file = Path(json_data['postprocess'].get('output_file'))
        thruster_outputs['output_path'] = output_file.relative_to(Path(output_path).resolve()).as_posix()

    return thruster_outputs
