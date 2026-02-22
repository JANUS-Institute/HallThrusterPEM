"""`mcmc.py`

This script is used to run MCMC on the cathode-thruster-plume system.

Requires a valid YAML config file specifying the system.
The outputs are written to a folder called amisc_{TIMESTAMP}/mcmc if no output directory is specified.
That folder in turn contains folders (numbered 000000 - num_samples) that hold the model outputs for each sample generated during MCMC.
Additionally, each sample, its log-pdf, and whether it was accepted are written to a file called mcmc.csv in the main mcmc directory.

For usage details and a full list of options , run 'pdm run scripts/run_mcmc.py --help'

"""  # noqa: E501

import argparse
from argparse import Namespace
import copy
import json
import math
import os
import pickle
import random
import shutil
import sys
import traceback
from dataclasses import fields, dataclass
from pathlib import Path
from typing import Callable, Type, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

import amisc.distribution as distributions

import hallmd.data
from hallmd.data import OperatingCondition, ThrusterData

import pem_mcmc as mcmc
from pem_mcmc.types import Dataset

from pem_core import PEM
from pem_core.types import Array, ArrayLike, PathLike, Variable
from pem_core.workflows.mcmc import _relative_gaussian_likelihood, _log_posterior
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis

parser = argparse.ArgumentParser(description="Run MCMC calibration for the Hall thruster PEM.")

parser.add_argument(
    "config_file",
    type=str,
    help="the path to the `amisc` YAML config file with model and input/output variable information.",
)

parser.add_argument(
    "--executor",
    type=str,
    default="process",
    help="the parallel executor for running MCMC samples. Options are `thread` or `process`. Default (`process`).",
)

parser.add_argument(
    "--max_workers",
    type=int,
    default=None,
    help="the maximum number of workers to use for parallel processing. Defaults to using maxnumber of available CPUs.",
)

parser.add_argument(
    "--max_samples",
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
    "--output_interval",
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

parser.add_argument("--output_dir", type=str, default=None, help="Directory into which output files are written")

parser.add_argument(
    "--init_sample",
    type=str,
    help="""CSV file containing initial sample. Only used with the `dram` sampler.""",
)

parser.add_argument(
    "--init_cov",
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


class Sampler:
    variables: list[Variable]
    base_vars: dict[Variable, Any]
    init_sample_file: PathLike | None
    init_cov_file: PathLike | None
    system: PEM

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        self.variables = variables
        self.data = data
        self.system = system
        self.base_vars = base_vars
        self.opts = opts
        self.init_sample_file = init_sample_file
        self.init_cov_file = init_cov_file
        self.logpdf = lambda x: _log_posterior(system, dict(zip(variables, x)), log_likelihood)
        self._init_sample = None
        self._init_cov = None

    def cov(self):
        return self.initial_cov()

    def initial_sample(self):
        if self._init_sample is None:
            # Read initial sample from file or create it from the base variables dict
            if self.init_sample_file is None:
                self._init_sample = np.array([self.base_vars[p] for p in self.variables])
            else:
                index_map = {p.name: i for (i, p) in enumerate(self.variables)}
                self._init_sample = np.zeros(len(self.variables))
                var_dict = mcmc.read_dlm(self.init_sample_file)
                for k, v in var_dict.items():
                    i = index_map[k]
                    self._init_sample[i] = v[0]

        return self._init_sample

    def initial_cov(self):
        if self._init_cov is None:
            index_map = {p.name: i for (i, p) in enumerate(self.variables)}
            if self.init_cov_file is None:
                variances = np.ones(len(self.variables))

                # Use variable distributions to estimate covariance
                for i, p in enumerate(self.variables):
                    dist = self.system.inputs()[p].distribution
                    if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                        lb, ub = dist.dist_args
                        std: float = (ub - lb) / 4
                    elif isinstance(dist, distributions.Normal):
                        std: float = dist.dist_args[1]
                    elif isinstance(dist, distributions.LogNormal):
                        std: float = dist.base ** dist.dist_args[1]
                    else:
                        raise ValueError(
                            f"Unsupported distribution {dist}. Currently only `Uniform`, `LogUniform`, `Normal` and `LogNormal` are supported."  # noqa: E501
                        )
                    # TODO: fix this typing issue
                    variances[i] = self.system.inputs()[p].normalize(std) ** 2  # type: ignore
                self._init_cov = np.diag(variances)
            else:
                # Construct covariance from file
                # We support the variables being in a different order so we build the covariance up from the index map.
                cov_dict = mcmc.read_dlm(self.init_cov_file)
                N = len(self.variables)
                self._init_cov = np.zeros((N, N))
                for i, (key1, column) in enumerate(cov_dict.items()):
                    i1 = index_map[key1]
                    for j, (var, key2) in enumerate(zip(column, cov_dict.keys())):
                        i2 = index_map[key2]
                        self._init_cov[i1, i2] = var

            # Verify that the covariance matrix is positive-definite before proceeding.
            # This throws an exception if not.
            np.linalg.cholesky(self._init_cov)
            self.init_cov = np.linalg.cholesky(self._init_cov)
        return self._init_cov


class PriorSampler(Sampler):
    """
    Samples from prior distribution, run model, and evaluates posterior probability
    """

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood, init_sample_file, init_cov_file)

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: fix this typing issue
        sample = np.array([var.normalize(var.distribution.sample((1,)))[0] for var in self.variables]) # type: ignore
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class PreviousRunSampler(Sampler):
    """
    Samples (with replacement) from a previous MCMC run.
    """

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        prev_run_file: PathLike,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
        burn_fraction: float = 0.5,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood,init_sample_file, init_cov_file)
        _vars, _samples, _, accepted, _ = mcmc.read_output_file(prev_run_file)

        # Figure out the first index we should sample.
        self.start_index = math.floor(burn_fraction * len(_samples))

        # associate variable names with columns and remove burned samples
        _samples = np.array(_samples)
        col_dict = {system.inputs()[v]: _samples[self.start_index :, i] for (i, v) in enumerate(_vars)}

        # reorder to match input varible list.
        # this will error if variable lists don't match
        self.samples = np.array([col_dict[v] for v in self.variables]).T
        self.accepted = accepted[self.start_index :]

    def __iter__(self):
        return self

    def sample_index(self):
        return random.randint(0, self.samples.shape[0] - 1)

    def __next__(self):
        # draw until we get an accepted sample
        index = self.sample_index()

        while not self.accepted[index]:
            index = self.sample_index()

        sample = self.samples[index, :]
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class DRAMSampler(Sampler):
    """
    Samples using delayed rejection adaptive metropolis
    """
    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood, init_sample_file, init_cov_file)

        self.sampler = DelayedRejectionAdaptiveMetropolis(
            self.logpdf,
            self.initial_sample(),
            self.initial_cov(),
            adapt_start=10,
            eps=1e-6,
            sd=None,
            interval=1,
            level_scale=1e-1,
        )

    # TODO: fix this typing issue
    def cov(self):  # type: ignore
        return self.sampler.cov_chol

    def __iter__(self):
        return iter(self.sampler)

@dataclass
class ExecutionOptions:
    executor: Type
    directory: Path
    max_workers: int
    fidelity: str | tuple | dict = (0, 0)
    noise_std: float = 0.05
    sample_aleatoric: bool = False
    print_likelihood: bool = True

def load_system_and_opts(args: Namespace) -> tuple[PEM, ExecutionOptions]:
    config = args.config_file
    system = PEM.from_file(config, output_dir=args.output_dir)
    system.set_logger(stdout=True)

    assert system.root_dir is not None, "System root directory must be set. This should have been done in PEM.from_yaml()."

    # Copy config file into output dir
    if Path(config).name not in os.listdir(system.root_dir):
        shutil.copy(config, system.root_dir)

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
        directory=system.root_dir / "mcmc",
        fidelity=(0, args.ncharge - 1),
        sample_aleatoric=args.sample_aleatoric,
        noise_std=args.noise_std,
    )

    os.mkdir(opts.directory)

    return system, opts

def _get_qoi_single_opcond(data: ThrusterData, observation: ThrusterData, field) -> tuple[Array, Array] | None:
    """
    For a specified field in the `ThrusterData` class, extract the value of that field.
    If the field is not present in both the data and observation, returns None.
    Otherwise, returns a tuple of the (data vals, obs. vals) where both are 1D numpy arrays of the same size.
    """

    data_field = getattr(data, field)
    obs_field = getattr(observation, field)

    if (data_field is None) or (obs_field is None):
        return None

    match field:
        case "ion_velocity":
            data_coords = data_field.axial_distance_m
            data_u = data_field.velocity_m_s.mean

            obs_coords = obs_field.axial_distance_m
            obs_u = obs_field.velocity_m_s.mean

            if obs_coords is None or obs_u is None:
                return None

            obs_u_itp = np.interp(data_coords, obs_coords, obs_u)

            return data_u, obs_u_itp

        case "ion_current_sweeps":
            data_j = []
            obs_j = []

            for data_sweep, obs_sweep in zip(data_field, obs_field):
                data_coords = data_sweep.angles_rad
                data_sweep_j = data_sweep.current_density_A_m2.mean

                obs_coords = obs_sweep.angles_rad
                obs_sweep_j = obs_sweep.current_density_A_m2.mean

                if obs_coords is None or obs_sweep_j is None:
                    return None

                obs_j_itp = np.interp(data_coords, obs_coords, obs_sweep_j)
                data_j.append(data_sweep_j)
                obs_j.append(obs_j_itp)

            data_j = np.concat(data_j)
            obs_j = np.concat(obs_j)

            return data_j, obs_j

        case _:
            return np.array([data_field.mean]), np.array([obs_field.mean])
        
def append_sample_row(
    logfile: PathLike, sample_index: int, sample: Array, logp: float, accepted: bool, delimiter: str = ','
):
    """Append a row of MCMC diagnostic data for the given `logfile`"""
    id_str = f"{sample_index:06d}"
    with open(logfile, "a") as fd:
        row = [id_str] + [f"{s}" for s in sample] + [f"{logp}", f"{accepted}"]
        print(delimiter.join(row), file=fd)


def _get_qoi_all_opconds(data: Dataset, observation: Dataset, field: str) -> tuple[Array, Array] | None:
    data_arrays = []
    obs_arrays = []
    """
    For a specified field in the `ThrusterData` class, extract all valid values of that field from both data and
    observation for all operating conditions where that field is present in both data and observation.
    If the field is not present in any operating condition, returns None.
    Otherwise, returns a tuple of the (data vals, obs. vals) where both are 1D numpy arrays of the same size.
    """

    for _data, _obs in zip(data.values(), observation.values()):
        out = _get_qoi_single_opcond(_data, _obs, field)
        if out is not None:
            data_arr, obs_arr = out
            obs_arrays.append(obs_arr)
            data_arrays.append(data_arr)

    if len(data_arrays) == 0 or len(obs_arrays) == 0:
        return None

    data_array_cat = np.concat(data_arrays)
    obs_array_cat = np.concat(obs_arrays)

    assert data_array_cat.size == obs_array_cat.size

    return data_array_cat, obs_array_cat

def _log_likelihood(
    data: Dataset,
    base_params: dict[str, ArrayLike],
    opts: ExecutionOptions,
) -> Callable[[PEM, dict[str, ArrayLike]], float]:
    
    def _likelihood_func(system: PEM, params: dict[str, ArrayLike]) -> float:
        result = _run_model(params, system, list(data.keys()), base_params, opts)

        if result is None:
            return -np.inf
        
        # Extract likelihood and L-2 distances for each component of the QoI (e.g. thrust, ion velocity, etc.)
        component_stats = {}

        L = 0.0
        for field in fields(ThrusterData):
                    # Assemble a vector of all data points for a single operating condition
            if (out := _get_qoi_all_opconds(data, result, field.name)) is None:
                continue

            data_arr, obs_arr = out
            distance, likelihood = _relative_gaussian_likelihood(data_arr, obs_arr, opts.noise_std)
            component_stats[field.name] = (distance, likelihood)
            L += likelihood

        # print table to output
        if opts.print_likelihood:
            field_names = [f.name for f in fields(ThrusterData)]
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
    operating_conditions: list[OperatingCondition],
    base_params: dict[str, ArrayLike],
    opts: ExecutionOptions,
) -> Dataset | None:
    sample_dict = copy.deepcopy(base_params)
    for key, val in params.items():
        sample_dict[key] = val

    # Add aleatoric variables if requested
    if opts.sample_aleatoric:
        sample_dict = _sample_aleatoric(sample_dict, system)

    output = None

    try:
        # Run model
        with opts.executor(max_workers=opts.max_workers) as exec:
            outputs = system.predict(
                sample_dict,
                use_model=opts.fidelity,
                model_dir=opts.directory,
                executor=exec,
                verbose=False,
            )

        # Get plume sweep radii
        sweep_radii = system['Plume'].model_kwargs['sweep_radius']

        # Assemble output dict from operating conditions -> results
        output_thrusterdata = hallmd.data.pem_to_thrusterdata(
            operating_conditions, outputs, sweep_radii, use_corrected_thrust=True
        )

        if output_thrusterdata is not None:
            # Write outputs to file
            with open(opts.directory / "pem.pkl", "wb") as fd:
                pickle.dump({"input": sample_dict, "output": output_thrusterdata}, fd)

        output = output_thrusterdata

    except Exception as e:
        print("Error detected. Failing input printed below")
        print(f"{sample_dict}")
        print(e)
        traceback.print_exc(file=sys.stdout)

    # Consolidate output files
    _consolidate_outputs(opts.directory)

    return output

def _sample_aleatoric(sample_dict, system):
    # We sample from normal distributions, as these are more reflective of the experimental uncertainty than
    # the uniform distributions that amisc uses for Relative.
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


def main(args):
    pem, opts = load_system_and_opts(args)
    base = pem.get_nominal_inputs(norm=True)
    params_to_calibrate = pem.get_inputs_by_category("calibration", sort="name")

    # Determine thruster and load data
    thruster_name = pem['Thruster'].model_kwargs['thruster']
    thruster = hallmd.data.get_thruster(thruster_name)
    datasets = thruster.datasets_from_names(args.datasets)
    data = hallmd.data.load(datasets)
    operating_conditions = list(data)

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a", "B_hat"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys[param.casefold()]
        inputs_unnorm = np.array([getattr(cond, long_name) for cond in data.keys()])

        if param == "B_hat" and param not in pem.inputs():
            base["B_hat"] = np.float64(1.0)
        else:
            p = pem.inputs()[param]
            base[param] = p.normalize(inputs_unnorm) # type: ignore

    # Set plume sweep radii based on data.
    # If there are no sweep radii in the data, we predict the current density
    # at a distance of 1 meter only.
    sweep_radii = []
    for d in data.values():
        if d.ion_current_sweeps is None:
            continue
        sweep_radii += [s.radius_m for s in d.ion_current_sweeps]
    if sweep_radii:
        sweep_radii = np.sort(np.unique(sweep_radii))
        pem['Plume'].model_kwargs['sweep_radius'] = np.array(sweep_radii)
    else:
        pem['Plume'].model_kwargs['sweep_radius'] = np.array([1.0])

    print(f"params_to_calibrate: {[p.name for p in params_to_calibrate]}")
    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Prepare output file
    root_dir = opts.directory
    logfile = root_dir / "mcmc.csv"
    delimiter = ","
    header = delimiter.join(["id"] + [p.name for p in params_to_calibrate] + ["log_posterior"] + ["accepted"])
    with open(logfile, "w") as fd:
        print(header, file=fd)

    # Set up directory for initial sample
    _update_opts(opts, root_dir, 0)

    # Set up samplers
    match args.sampler:
        case "dram":
            sampler = DRAMSampler(
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
            sampler = PriorSampler(params_to_calibrate, data, pem, base, opts, _log_likelihood(data, base, opts))
        case "prev-run":
            sampler = PreviousRunSampler(
                params_to_calibrate, data, pem, base, opts,
                prev_run_file=args.prev, burn_fraction=0.5, log_likelihood=_log_likelihood(data, base, opts)
            )
        case _:
            raise ValueError("Unreachable")

    # Initialize sampler parameters and evaluate posterior on init sample
    max_samples: int = args.max_samples
    init_sample = sampler.initial_sample()
    best_logp = sampler.logpdf(init_sample)
    append_sample_row(logfile, 0, init_sample, best_logp, True)
    num_accept = 1

    if isinstance(sampler, DRAMSampler):
        burn_fraction = 0.5
        num_subsample = 1000
    else:
        burn_fraction = 0.0
        num_subsample = None

    # Main sampler loop
    start_index = 1
    _update_opts(opts, root_dir, start_index)
    for i, (sample, logp, accepted_bool) in enumerate(sampler):
        append_sample_row(logfile, start_index + i, sample, logp, accepted_bool)

        if logp > best_logp:
            best_logp = logp

        if accepted_bool:
            num_accept += 1

        print(
            f"sample: {i + start_index}/{max_samples}, logp: {logp:.3f}, best logp: {best_logp:.3f},",
            f"accepted: {accepted_bool}, p_accept: {num_accept / (i + 1 + start_index) * 100:.1f}%",
        )

        corner = i == max_samples or (i > 100 and (i % (args.output_interval * 10) == 0))

        if (i == max_samples) or (i % args.output_interval == 0):
            mcmc.analyze(
                root_dir.parent,
                args.datasets,
                plot_corner=corner,
                plot_bands=True,
                plot_traces=True,
                calc_metrics=False,
                save_restart=True,
                proposal_cov=sampler.cov(),
                subsample=num_subsample,
                burn_fraction=burn_fraction,
            )

        if i >= max_samples:
            break

        _update_opts(opts, root_dir, i + 1 + start_index)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
