"""`mcmc.py`

This script is used to run MCMC on the cathode-thruster-plume system.

Requires a valid YAML config file specifying the system.
The outputs are written to a folder called amisc_{TIMESTAMP}/mcmc if no output directory is specified.
That folder in turn contains folders (numbered 000000 - num_samples) that hold the model outputs for each sample generated during MCMC.
Additionally, each sample, its log-pdf, and whether it was accepted are written to a file called mcmc.csv in the main mcmc directory.

For usage details and a full list of options , run 'pdm run scripts/run_mcmc.py --help'

"""  # noqa: E501

#%%
import argparse
from argparse import Namespace
import copy
import json
import os
import pickle
import shutil
import sys
import traceback
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Type, cast, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import xarray as xr

import amisc.distribution as distributions

from pem_core import PEM
from pem_core.data import UNITS, load_multiple_datasets, DataEntry, DataInstance, DataField, interpolate_data_instance
from pem_core.types import ArrayLike
from pem_core.sampling import _relative_gaussian_likelihood
import pem_core.sampling as sampling

parser = argparse.ArgumentParser(description="Run MCMC calibration for the Hall thruster PEM.")

parser.add_argument(
    "config",
    type=str,
    help="the path to the `amisc` YAML config file with model and input/output variable information.",
)

parser.add_argument(
    "--executor",
    type=str,
    choices=["process", "thread"],
    default="process",
    help="the parallel executor for running MCMC samples.",
)

parser.add_argument(
    "--max-workers",
    type=int,
    default=None,
    help="the maximum number of workers to use for parallel processing. Defaults to using maxnumber of available CPUs.",
)

parser.add_argument(
    "--max-samples",
    type=int,
    default=1,
    help="The maximum number of samples to generate using MCMC",
)

parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=["diamant2014", "macdonald2019"],
    help="""
    A list of datasets to use.
    For the SPT-100, pick from [diamant2014, macdonald2019, sankovic1993]".
    For the H9, pick from [um2024, gt2024].
    """,
)

parser.add_argument(
    "--output-interval",
    type=int,
    default=100,
    help="How frequently plots are generated",
)

parser.add_argument(
    "--ncharge",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="Number of ion charge states to include in simulation. Must be between 1 and 3.",
)

parser.add_argument(
    "--noise-std",
    type=float,
    default=0.05,
    help="Standard deviation of the Gaussian noise in the likelihood calculation",
)

parser.add_argument("--output-dir", type=str, default=None, help="Directory into which output files are written")

parser.add_argument(
    "--init-sample",
    type=str,
    help="""CSV file containing initial sample. Only used with the `dram` sampler.""",
)

parser.add_argument(
    "--init-cov",
    type=str,
    help="""CSV file containing initial covariance matrix. Only used with the `dram` sampler.""",
)

parser.add_argument(
    "--sampler",
    type=str,
    choices=["dram", "prior", "prev-run", "fixed"],
    default="dram",
    help="""
    The type of sampler to use
    - `dram`: the delayed-rejection adaptive metropolis sampler
    - `prior`: sample from the variable prior distributions only
    - `prev-run`: draw randomly (with replacement) from the samples of a previous run.
    - `fixed` : evaluates only the initial sample

    The `prev-run` sampler requires the --prev argument to be passed.
    """,
)

parser.add_argument(
    "--prev",
    type=str,
    help="""
    The file containing samples from a previous run using this config.
    This is only useful when using the `prev-run` sampler.
    """,
)

parser.add_argument(
    "--sample-aleatoric",
    action='store_true',
    help="Sample aleatoric varibles from distributions in config file instead of using fixed values.",
)

#%%
@dataclass
class ExecutionOptions:
    executor: Type
    directory: Path
    max_workers: int
    fidelity: str | tuple | dict = (0, 0)
    noise_std: float = 0.05
    sample_aleatoric: bool = False
    print_likelihood: bool = True

def load_pem_and_opts(args: Namespace) -> tuple[PEM, ExecutionOptions]:
    config = args.config
    pem = PEM.from_file(config, output_dir=args.output_dir)
    pem.set_logger(stdout=True)

    assert pem.root_dir is not None, "PEM root directory must be set. This should have been done in PEM.from_yaml()."

    # Copy config file into output dir
    if Path(config).name not in os.listdir(pem.root_dir):
        shutil.copy(config, pem.root_dir)

    match args.executor.lower():
        case "thread":
            executor = ThreadPoolExecutor
        case "process":
            executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    opts = ExecutionOptions(
        executor,
        max_workers=args.max_workers,
        directory=pem.root_dir / "mcmc",
        fidelity=(0, args.ncharge - 1),
        sample_aleatoric=args.sample_aleatoric,
        noise_std=args.noise_std,
    )

    os.makedirs(opts.directory, exist_ok=True)
    return pem, opts

def _sample_aleatoric(sample_dict, system):
    """
    Sample the aleatoric variables using nomral distributions, rather than the uniform distributions that amisc uses for Relative
    """
    aleatoric_vars = ["P_b", "V_a", "mdot_a"]
    for var in aleatoric_vars:
        nominal = sample_dict[var]
        input = system.inputs()[var]
        dist = input.distribution
        assert isinstance(dist, distributions.Relative)

        # we assume the error specified in the config file is 1 standard deviation
        percent = dist.dist_args[0] / 100

        for i in range(np.atleast_1d(nominal).size):
            nom = input.denormalize(nominal[i])
            sample = nom * (1 + percent * np.random.randn())
            nominal[i] = input.normalize(sample)

        sample_dict[var] = nominal
    return sample_dict

def _update_opts(opts, root_dir, sample_index):
    id_str = f"{sample_index:06d}"
    opts.directory = root_dir / id_str
    os.mkdir(opts.directory)

def pem_to_xarray(
        operating_conditions: list[dict[str, float]],
        outputs: dict, sweep_radii: np.ndarray,
        use_corrected_thrust: bool = True
    ) -> list[DataEntry]:

    """
    Convert the outputs of the Hall thruster PEM to xarrays to be compared to data
    """
    data_entries: list[DataEntry] = []

    for (i, opcond) in enumerate(operating_conditions):

        if use_corrected_thrust:
            # With multiple radii, we have multiple thrust. Pick the last one as sweep_radii are sorted
            thrust = xr.DataArray(np.atleast_1d(outputs['T_c'][i])[-1])
        else:
            thrust = xr.DataArray(outputs['T'][i])

        Id = xr.DataArray(outputs['I_d'][i])
        Vcc = xr.DataArray(outputs['V_cc'][i])

        z = outputs['u_ion_coords'][i]
        uion = outputs['u_ion'][i]
        uion_arr = xr.DataArray(uion, coords=[z], dims=["z"])

        theta = outputs['j_ion_coords'][i]
        r = sweep_radii
        jion = np.atleast_3d(outputs['j_ion'])[i, :, :].T
        jion_arr = xr.DataArray(jion, coords=[r, theta], dims=["r", "theta"])

        instance: DataInstance = {
            "discharge current": DataField(val=Id, unit="A"),
            "cathode coupling voltage": DataField(val=Vcc, unit="V"),
            "thrust": DataField(val=thrust, unit="N"),
            "ion velocity": DataField(val=uion_arr, unit="m/s"),
            "ion current density": DataField(val=jion_arr, unit="A/m^2"),
        }

        entry = DataEntry(operating_condition=opcond, data=instance)
        data_entries.append(entry)

    return data_entries

def _consolidate_outputs(path: Path):
    """
    By default, amisc creates a directory with Thruster, Cathode, and Plume folders. The latter two are empty, but the
    former has individual files for each individual Hall thruster simulation. This creates a ton of individual files
    that can quickly deplete the maximum allowance of the Great Lakes scratch partition. This script deletes the empty
    folders, consolidates all output files into a single file, and deletes the individual files. Additionally, for
    failed runs the containing folder may be empty, so we delete it.
    """

    # Check if folder is empty
    if not os.listdir(path):
        shutil.rmtree(path)

    cathode_path = path / "Cathode"
    plume_path = path / "Plume"
    thruster_path = path / "Thruster"

    if thruster_path.is_dir():
        jsons = [f for f in thruster_path.iterdir() if f.suffix == ".json"]

        # Read JSON files and consolidate them into a single file
        json_contents = []
        for json_file in jsons:
            with open(json_file, "r") as fd:
                contents = json.load(fd)
                json_contents.append(contents)

        with open(path / "thruster.json", "w") as fd:
            json.dump(json_contents, fd)

    # Delete unneeded files
    if cathode_path.is_dir():
        cathode_path.rmdir()

    if plume_path.is_dir():
        plume_path.rmdir()

    if thruster_path.is_dir():
        shutil.rmtree(thruster_path)

def _run_model(
    params: dict[str, ArrayLike],
    system: PEM,
    operating_conditions: list[dict[str, float]],
    base_params: dict[str, ArrayLike],
    opts: ExecutionOptions,
) -> list[DataEntry] | None:
    """
    Run the Hall thruster PEM at a list of operating conditions for a set of calibration params (params), leaving the remaining parameters equal to the values in `base_params`
    """
    # Start with nominal values for all parameters, and then update with the current sample values.
    sample_dict = copy.deepcopy(base_params)
    for key, val in params.items():
        sample_dict[key] = val

    # Add aleatoric variables if requested
    if opts.sample_aleatoric:
        sample_dict = _sample_aleatoric(sample_dict, system)

    output = None

    # Run model
    with opts.executor(max_workers=opts.max_workers) as exec:
        outputs = system.predict(
            sample_dict,
            use_model=opts.fidelity,
            model_dir=opts.directory,
            executor=exec,
            verbose=False,
        )

    if "errors" in outputs:
        # Rethrow error
        raise ChildProcessError(outputs["errors"][0]['error'])

    # Get plume sweep radii
    if "Plume" in [c.name for c in system.components]:
        sweep_radii = system['Plume'].model_kwargs['sweep_radius']
    else:
        sweep_radii = np.array([1.0])


    output_thrusterdata = pem_to_xarray(operating_conditions, cast(dict, outputs), sweep_radii, use_corrected_thrust=True)

    if output_thrusterdata is not None:
        # Write outputs to file
        with open(opts.directory / "pem.pkl", "wb") as fd:
            pickle.dump({"input": sample_dict, "output": output_thrusterdata}, fd)

        output = output_thrusterdata

    # Consolidate output files
    _consolidate_outputs(opts.directory)

    return output

def _log_likelihood(
    data: list[DataEntry],
    base_params: dict[str, ArrayLike],
    opts: ExecutionOptions,
) -> Callable[[PEM, dict[str, ArrayLike]], float]:
    """
    Return a log-likelihood function comparing simulation outputs to data
    The likelihood function has signature likeliood(pem, params) -> float
    """
    def _likelihood_func(system: PEM, params: dict[str, ArrayLike]) -> float:
        opconds = [d.operating_condition for d in data]
        sim_results = _run_model(params, system, opconds, base_params, opts)

        if sim_results is None:
            return -np.inf
        
        # Extract likelihood and L-2 distances for each component of the QoI (e.g. thrust, ion velocity, etc.)
        component_stats = {}

        # Interpolate sim results to data coordinates
        sim_itp = [interpolate_data_instance(_sim.data, _data.data) for (_sim, _data) in zip(sim_results, data)]

        # Extract data per-field into 1-D vectors
        sim_vectors = {}
        data_vectors = {}
        data_errs = {}

        L = 0.0

        field_names = set()
        for _data in data:
            field_names.update(list(_data.data.keys()))
        field_names = list(field_names)

        for field_name in field_names:
            sim_vectors[field_name] = []
            data_vectors[field_name] = []
            data_errs[field_name] = []

            for _sim, _data in zip(sim_itp, data):
                if field_name not in _data.data:
                    continue
                data_field = _data.data[field_name]
                sim_field = _sim[field_name]

                sim_vectors[field_name].append(sim_field.val.values.flatten())
                data_vectors[field_name].append(data_field.val.values.flatten())

                if data_field.err is not None:
                    data_errs[field_name].append(data_field.err.values.flatten())

            sim_vec = np.concatenate(sim_vectors[field_name])
            data_vec = np.concatenate(data_vectors[field_name])
            distance, likelihood = _relative_gaussian_likelihood(data_vec, sim_vec, opts.noise_std)        

            component_stats[field_name] = (distance, likelihood)
            L += likelihood

        # print table to output
        if opts.print_likelihood:
            name_len = max(len(n) for n in field_names) + 1
            indent = "  "

            header = indent + " Field" + (" " * (name_len - 5)) + "|  L2 norm   | Likelihood  "
            rule = "-" * len(header)

            print("\n" + rule)
            print(header)
            print(rule)

            for name, (l2, logp) in component_stats.items():
                pad = name_len - len(name)
                print(indent + f" {name}{' ' * pad}| {l2:10.5f} | {logp:10.5f}")

            print(rule + "\n")

        return L

    return _likelihood_func


UNITS.define("Torr = 133.322368 pascal = Torr")

operating_vars = {
    "discharge voltage": {
        "unit": UNITS.volts,
    },
    "anode mass flow rate": {
        "unit": UNITS.kg / UNITS.second,
    },
    "background pressure": {
        "unit": UNITS.torr,
        "default": 0.0,
    },
    "magnetic field scale": {
        "unit": UNITS.dimensionless,
        "default": 1.0,
    },
}

coords = {
    "z": UNITS.meter,
    "r": UNITS.meter,
    "theta": UNITS.rad,
}

qois = {
    "cathode coupling voltage": {
        "unit": UNITS.volts,
    },
    "discharge current": {
        "unit": UNITS.ampere,
    },
    "thrust": {
        "unit": UNITS.newton,
    },
    "ion velocity": {
        "unit": UNITS.meter / UNITS.second,
        "coords": ("z",),
    },
    "ion current density": {
        "unit": UNITS.ampere / UNITS.meter**2,
        "coords": ("r", "theta"),
    },
}

FLOW_RATE_KEY = "anode mass flow rate"

rename_map = {
    "anode voltage" : "discharge voltage",
    "anode current" : "discharge current",
    "anode flow rate" : FLOW_RATE_KEY,
    "axial distance from anode": "z",
    "axial position from anode": "z",
    "axial ion velocity": "ion velocity",
    "angular position from thruster centerline": "theta",
    "radial position from thruster exit": "r",
}

var_names = {
    "discharge voltage": "V_a",
    "anode mass flow rate": "mdot_a",
    "background pressure": "P_b",
    "magnetic field scale": "B_hat",
}

def main(args):
    # Load PEM from YAML file and determine nominal and calibration parameters
    pem, opts = load_pem_and_opts(args)
    component_names = set([c.name for c in pem.components])
    base = pem.get_nominal_inputs(norm=True)
    params_to_calibrate = pem.get_inputs_by_category("calibration", sort="name")


    # Load data from files
    data = load_multiple_datasets(args.datasets, operating_vars, qois, coords, rename_map=rename_map)
    operating_conditions = [d.operating_condition for d in data]

    # Load operating conditions into input dictionary
    for (long_name, param) in var_names.items():
        if param not in pem.inputs():
            base[param] = 1.0
            continue

        inputs_unnorm = np.array([cond[long_name] for cond in operating_conditions])
        p = pem.inputs()[param]
        base[param] = p.normalize(inputs_unnorm) # type: ignore

    # Remove data where theta < 0 for ion current density
    for d in data:
        if 'ion current density' not in d.data:
            continue
        jion_val = d.data['ion current density'].val
        jion_val = jion_val.sel(theta=slice(0, np.pi/2))

        jion_err = d.data['ion current density'].err
        if jion_err is not None:
            jion_err = jion_err.sel(theta=slice(0, np.pi/2))

        d.data['ion current density'].val = jion_val
        d.data['ion current density'].err = jion_err

    # Set plume sweep radii based on data.
    # If there is no ion current density in the data, we just set the sweep radius to 1.0 (i.e. no sweeping). This allows us to still run MCMC on cases where there is no ion current density data without having to change the model config.
    if "Plume" in component_names:
        sweep_radii = []
        for entry in data:
            entry_data = entry.data
            if "ion current density" not in entry_data:
                continue

            ji_coords = entry_data["ion current density"].val.coords
            if ji_coords is None or "r" not in ji_coords:
                continue

            sweep_radii.extend(ji_coords["r"].values)

        if sweep_radii:
            sweep_radii = np.sort(np.unique(sweep_radii))
            pem['Plume'].model_kwargs['sweep_radius'] = np.array(sweep_radii).astype(float)
        else:
            pem['Plume'].model_kwargs['sweep_radius'] = np.array([1.0])

    print(f"params_to_calibrate: {[p.name for p in params_to_calibrate]}")
    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Set up directory for initial sample
    # TODO: _update_opts() should probably be baked into the samplers/loggers
    root_dir = opts.directory
    sample_dir = opts.directory / "samples"
    os.makedirs(sample_dir, exist_ok=True)
    _update_opts(opts, sample_dir, 0)

    # Set up samplers
    match args.sampler:
        case "dram":
            sampler = sampling.DRAMSampler(
                params_to_calibrate,
                data,
                pem,
                base,
                opts,
                _log_likelihood(data, base, opts),
                init_sample_file=args.init_sample,
                init_cov_file=args.init_cov,
            )
        case "prior":
            sampler = sampling.PriorSampler(params_to_calibrate, data, pem, base, opts, _log_likelihood(data, base, opts))
        case "prev-run":
            sampler = sampling.PreviousRunSampler(
                params_to_calibrate, data, pem, base, opts,
                prev_run_file=args.prev, burn_fraction=0.5, log_likelihood=_log_likelihood(data, base, opts)
            )
        case _:
            raise ValueError("Unreachable")

    # Initialize sampler parameters and evaluate posterior on init sample
    mcmc_logger = sampling.SampleLogger(sampler, output_dir=root_dir)
    _update_opts(opts, sample_dir, mcmc_logger.sample_index+1)

    for i, _ in itertools.islice(enumerate(sampler), args.max_samples):
        mcmc_logger.update(log=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
