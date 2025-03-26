"""`mcmc.py`

This script is used to run MCMC on the cathode-thruster-plume system.

Requires a valid YAML config file specifying the system.
The outputs are written to a folder called amisc_{TIMESTAMP}/mcmc if no output directory is specified.
That folder in turn contains folders (numbered 000000 - num_samples) that hold the model outputs for each sample generated during MCMC.
Additionally, each sample, its log-pdf, and whether it was accepted are written to a file called mcmc.csv in the main mcmc directory.

For usage details and a full list of options , run 'pdm run scripts/mcmc.py --help'

"""  # noqa: E501

import argparse
import os
import pickle
import random
import shutil
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Type, TypeAlias, TypeVar

import amisc.distribution as distributions
import mcmc_analysis as analysis
import numpy as np
from amisc import Component, System, Variable, YamlLoader
from MCMCIterators import samplers

import hallmd.data
from hallmd.data import Array, OperatingCondition, PathLike, ThrusterData

Value: TypeAlias = np.float64 | Array
NaN = np.float64(np.nan)
Inf = np.float64(np.inf)

T = TypeVar("T", np.float64, Array)

parser = argparse.ArgumentParser(description="MCMC scripts")

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
    choices=["dram", "prior", "prev-run"],
    default="dram",
    help="""
    The type of sampler to use
    - `dram`: the delayed-rejection adaptive metropolis sampler
    - `prior`: sample from the variable prior distributions only
    - `prev-run`: draw randomly (with replacement) from the samples of a previous run.

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


@dataclass
class ExecutionOptions:
    executor: Type
    max_workers: int
    directory: Path
    fidelity: str | tuple | dict = (0, 0)
    use_plume: bool = True
    use_cathode: bool = True
    sample_aleatoric: bool = False


def load_system_and_opts(args: Namespace) -> tuple[System, ExecutionOptions]:
    config = args.config_file
    system = YamlLoader.load(config)
    if args.output_dir is None:
        system.root_dir = Path(config).parent
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        system.root_dir = Path(args.output_dir)

    print(system.root_dir)

    system.set_logger(stdout=True)

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
    )

    os.mkdir(opts.directory)

    return system, opts


def get_nominal_inputs(system: System) -> dict[str, Value]:
    """Create a dict mapping system inputs to their nominal values"""
    inputs: dict[str, Value] = {}
    for input in system.inputs():
        value = input.normalize(input.get_nominal())
        inputs[input.name] = value

    return inputs


def _get_qoi_single_opcond(data: ThrusterData, observation: ThrusterData, field) -> Optional[tuple[Array, Array]]:
    data_field = getattr(data, field)
    obs_field = getattr(observation, field)
    """
    For a specified field in the `ThrusterData` class, extract the value of that field.
    If the field is not present in both the data and observation, returns None.
    Otherwise, returns a tuple of the (data vals, obs. vals) where both are 1D numpy arrays of the same size.
    """

    if data_field is None or obs_field is None:
        return None

    match field:
        case "ion_velocity":
            data_coords = data_field.axial_distance_m
            data_u = data_field.velocity_m_s.mean

            obs_coords = obs_field.axial_distance_m
            obs_u = obs_field.velocity_m_s.mean

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

                obs_j_itp = np.interp(data_coords, obs_coords, obs_sweep_j)
                data_j.append(data_sweep_j)
                obs_j.append(obs_j_itp)

            data_j = np.concat(data_j)
            obs_j = np.concat(obs_j)

            return data_j, obs_j

        case _:
            return np.array([data_field.mean]), np.array([obs_field.mean])


def _get_qoi_all_opconds(
    data: dict[OperatingCondition, ThrusterData], observation: dict[OperatingCondition, ThrusterData], field: str
) -> Optional[tuple[Array, Array]]:
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


def _relative_l2_norm(data: Array, observation: Array) -> np.float64:
    """
    Compute the L2-normed distance between a data array and observation array,
    relative to the magnitude of the data array.
    """
    num = np.sum((data - observation) ** 2)
    denom = np.sum(data**2)
    return np.sqrt(num / denom)


def _gauss_logpdf_1D(x: np.float64 | float, mean: np.float64 | float, std: np.float64 | float) -> np.float64 | float:
    """
    Gaussian log-likelihood in 1D
    """
    return -0.5 * (np.log(2 * np.pi * std**2) + (x - mean) ** 2 / std**2)


def _sample_aleatoric(sample_dict, system):
    aleatoric_vars = ["P_b", "V_a", "mdot_a"]
    for var in aleatoric_vars:
        nominal = sample_dict[var]
        for i in range(np.atleast_1d(nominal).size):
            input = system.inputs()[var]
            nom = input.denormalize(nominal[i])
            nominal[i] = input.normalize(input.distribution.sample((1,), nom))[0]

        sample_dict[var] = nominal


def _run_model(
    params: dict[str, Value],
    system: System,
    operating_conditions: list[OperatingCondition],
    base_params: dict[str, Value],
    opts: ExecutionOptions,
):
    sample_dict = base_params.copy()
    for key, val in params.items():
        sample_dict[key] = val

    # Add aleatoric variables if requested
    if opts.sample_aleatoric:
        _sample_aleatoric(sample_dict, system)

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
            operating_conditions, outputs, sweep_radii, use_corrected_thrust=opts.use_plume
        )

        # Write outputs to file
        with open(opts.directory / "pem.pkl", "wb") as fd:
            pickle.dump({"input": sample_dict, "output": output_thrusterdata}, fd)

        return output_thrusterdata

    except Exception as e:
        print("Error detected. Failing input printed below")
        print(f"{sample_dict}")
        print(e)
        return None


def log_prior(system: System, params: dict[str, Value]) -> np.float64:
    logp = 0.0
    for key, value in params.items():
        var = system.inputs()[key]
        prior = var.distribution.pdf(var.denormalize(value))
        if isinstance(prior, np.ndarray):
            prior = prior[0]

        if prior <= 0:
            return -Inf

        logp += np.log(prior)

    return np.float64(logp)


def log_likelihood(
    params: dict[str, Value],
    data: dict[OperatingCondition, ThrusterData],
    system: System,
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> np.float64 | float:
    result = _run_model(params, system, list(data.keys()), base_params, opts)
    if result is None:
        return -np.inf

    L = 0.0
    std = 0.05  # assume 5% relative error

    component_stats = {}

    field_names = [f.name for f in fields(ThrusterData)]

    for field in field_names:
        # Assemble a vector of all data points for a single operating condition
        out = _get_qoi_all_opconds(data, result, field)
        if out is None:
            continue

        data_arr, obs_arr = out
        distance = _relative_l2_norm(data_arr, obs_arr)

        # Since the l2-norm is positive-definite, the distribution is a half-normal, so we
        # need to multiply the normal pdf by 2 (or add ln(2) to the log-pdf)
        likelihood = _gauss_logpdf_1D(distance, 0.0, std) + np.log(2)

        component_stats[field] = (distance, likelihood)

        L += likelihood

    # print table to output
    print_table = True
    if print_table:
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

    # if somehow nan or otherwise non-finite, return -inf
    if not np.isfinite(L) or np.isnan(L):
        return -np.inf

    return L


def log_posterior(
    params: dict[str, Value],
    data: dict[OperatingCondition, ThrusterData],
    system: System,
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> np.float64:
    prior = log_prior(system, params)
    if not np.isfinite(prior):
        return -Inf

    likelihood = log_likelihood(params, data, system, base_params, opts)
    if not np.isfinite(likelihood):
        return -Inf

    return prior + likelihood


def update_opts(root_dir, sample_index):
    id_str = f"{sample_index:06d}"
    opts.directory = root_dir / id_str
    os.mkdir(opts.directory)


def append_mcmc_diagnostic_row(logfile, sample_index, sample, logp, accepted_bool):
    """Append a row of MCMC diagnostic data for the given `logfile`"""
    id_str = f"{sample_index:06d}"
    with open(logfile, "a") as fd:
        row = [id_str] + [f"{s}" for s in sample] + [f"{logp}", f"{accepted_bool}"]
        print(delimiter.join(row), file=fd)


def read_dlm(file: PathLike, delimiter: str | None = ',', comments='#') -> dict[str, Array]:
    """Read a simple delimited file consisting of headers and numerical data into a dict that maps names to columns"""
    with open(file, 'r') as fd:
        header = fd.readline().rstrip()
        if header.startswith(comments):
            header = header[1:].lstrip()

    col_names = header.split(delimiter)
    table_data = np.atleast_2d(np.genfromtxt(file, skip_header=1, delimiter=delimiter))
    columns = [table_data[:, i] for i in range(len(col_names))]
    return {col_name: column for (col_name, column) in zip(col_names, columns)}


def get_calibration_params(component: Component, sort: str | None = None) -> list[Variable]:
    """Extract calibration params from the provided `component`.
    Optionally sort them based on either their 'name' or 'tex' representations"""

    params = [p for p in component.inputs if p.category == "calibration"]
    if sort is None:
        return params

    names = [getattr(p, sort).casefold() for p in params]
    sorted_params = [p for _, p in sorted(zip(names, params))]
    return sorted_params


class PreviousRunSampler:
    """
    Samples (with replacement) from a previous MCMC run
    """

    def __init__(self, file, variables, data, system, base_vars, opts, burn_fraction=0.5):
        self.variables = variables
        self.data = data
        self.system = system
        self.base_vars = base_vars
        self.opts = opts

        _vars, _samples, *_ = analysis.read_output_file(Path(file))
        self.start_index = round(burn_fraction * len(_samples))

        # associate variable names with columns and remove burned samples
        _samples = np.array(_samples)
        col_dict = {system.inputs[v]: _samples[start_index - 1 :, i] for (i, v) in enumerate(_vars)}

        # reorder to match input varible list.
        # this will error if variable lists don't match
        self.samples = np.array(col_dict[v] for v in self.variables).T

    def __iter__(self):
        return self

    def __next__(self):
        print(f"{self.samples.shape=}")

        index = random.randint(0, self.samples.shape[0] - 1)

        sample = self.samples[index, :]
        logp = log_posterior(dict(zip(self.variables, sample)), self.data, self.system, self.base_vars, self.opts)
        accept = True
        return (sample, logp, accept)


class PriorSampler:
    """
    Samples from prior distribution, run model, and evaluates posterior probability
    """

    def __init__(self, variables, data, system, base_vars, opts):
        self.variables = variables
        self.data = data
        self.system = system
        self.base_vars = base_vars
        self.opts = opts

    def __iter__(self):
        return self

    def __next__(self):
        sample = np.array([var.normalize(var.distribution.sample((1,)))[0] for var in self.variables])
        logp = log_posterior(dict(zip(self.variables, sample)), self.data, self.system, self.base_vars, self.opts)
        accept = True
        return (sample, logp, accept)


if __name__ == "__main__":
    args = parser.parse_args()
    system, opts = load_system_and_opts(args)
    base = get_nominal_inputs(system)

    # Determine thruster and load data
    thruster_name = system['Thruster'].model_kwargs['thruster']
    thruster = hallmd.data.get_thruster(thruster_name)
    datasets = thruster.datasets_from_names(args.datasets)
    data = hallmd.data.load(datasets)
    operating_conditions = list(data.keys())

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys[param.casefold()]
        inputs_unnorm = np.array([getattr(cond, long_name) for cond in data.keys()])
        base[param] = system.inputs()[param].normalize(inputs_unnorm)

    # Choose calibration params
    # Only calibrate cathode coupling params if we have cathode data, and plume params if we have plume data
    # Always include thruster params
    opts.use_cathode = any(d.cathode_coupling_voltage_V is not None for d in data.values())
    opts.use_plume = any(d.ion_current_sweeps is not None for d in data.values())

    sortmethod = 'name'
    cathode_params = get_calibration_params(system['Cathode'], sort=sortmethod) if opts.use_cathode else []
    plume_params = get_calibration_params(system['Plume'], sort=sortmethod) if opts.use_plume else []
    thruster_params = get_calibration_params(system['Thruster'], sort=sortmethod)
    # assemble params_to_calibrate, has to be done in this way to avoid duplicate Te in cathode and thruster
    params_to_calibrate: list[Variable] = cathode_params
    params_to_calibrate += [p for p in thruster_params if p not in params_to_calibrate]
    params_to_calibrate += [p for p in plume_params if p not in params_to_calibrate]

    # Set plume sweep radii based on data
    if opts.use_plume:
        sweep_radii = []
        for d in data.values():
            if d.ion_current_sweeps is None:
                continue
            sweep_radii += [s.radius_m for s in d.ion_current_sweeps]
        sweep_radii = np.sort(np.unique(sweep_radii))
        system['Plume'].model_kwargs['sweep_radius'] = sweep_radii

    print(f"{params_to_calibrate=}")
    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Prepare output file
    root_dir = opts.directory
    logfile = root_dir / "mcmc.csv"
    delimiter = ","
    header = delimiter.join(["id"] + [p.name for p in params_to_calibrate] + ["log_posterior"] + ["accepted"])
    with open(logfile, "w") as fd:
        print(header, file=fd)

    # Initialize sampler parameters
    max_samples: int = args.max_samples
    best_sample = None
    best_logp = -np.inf
    num_accept = 0
    start_index = 0

    # Set up samplers
    match args.sampler:
        case "dram":
            # Create initial sample
            index_map = {p.name: i for (i, p) in enumerate(params_to_calibrate)}
            init_sample = np.array([base[p] for p in params_to_calibrate])
            if args.init_sample is not None:
                var_dict = read_dlm(args.init_sample)
                for k, v in var_dict.items():
                    i = index_map[k]
                    init_sample[i] = v[0]

            # Create initial covariance
            if args.init_cov is None:
                variances = np.ones(len(params_to_calibrate))

                for i, p in enumerate(params_to_calibrate):
                    dist = system.inputs()[p].distribution
                    if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                        lb, ub = dist.dist_args
                        std = (ub - lb) / 4
                    elif isinstance(dist, distributions.Normal):
                        std = dist.dist_args[1]
                    elif isinstance(dist, distributions.LogNormal):
                        std = dist.base ** dist.dist_args[1]
                    else:
                        raise ValueError(f"Invalid distribution {dist}")

                    variances[i] = system.inputs()[p].normalize(std) ** 2

                init_cov = np.diag(variances)
            else:
                cov_dict = read_dlm(args.init_cov)
                N = len(params_to_calibrate)
                init_cov = np.zeros((N, N))
                for i, (key1, column) in enumerate(cov_dict.items()):
                    i1 = index_map[key1]
                    for j, (var, key2) in enumerate(zip(column, cov_dict.keys())):
                        i2 = index_map[key2]
                        init_cov[i1, i2] = var

            # Verify that the covariance matrix is positive-definite before proceeding
            np.linalg.cholesky(init_cov)

            # Generate initial sample
            update_opts(root_dir, start_index)
            logpdf = lambda x: log_posterior(dict(zip(params_to_calibrate, x)), data, system, base, opts)
            best_sample = init_sample
            init_logp = logpdf(init_sample)
            start_index += 1
            num_accept += 1
            print(f"{init_logp=}")
            append_mcmc_diagnostic_row(logfile, 0, init_sample, init_logp, True)

            sampler = samplers.DelayedRejectionAdaptiveMetropolis(
                logpdf,
                init_sample,
                init_cov,
                adapt_start=10,
                eps=1e-6,
                sd=None,
                interval=1,
                level_scale=1e-1,
            )
        case "prior":
            sampler = PriorSampler(params_to_calibrate, data, system, base, opts)
        case "prev-run":
            sampler = PreviousRunSampler(args.prev, params_to_calibrate, data, system, base, opts)
        case _:
            raise ValueError("Unreachable")

    update_opts(root_dir, start_index)

    # Main sampler loop
    for i, (sample, logp, accepted_bool) in enumerate(sampler):
        append_mcmc_diagnostic_row(logfile, i + start_index, sample, logp, accepted_bool)

        if logp > best_logp:
            best_sample = sample
            best_logp = logp

        if accepted_bool:
            num_accept += 1

        print(
            f"sample: {i + start_index}/{max_samples}, logp: {logp:.3f}, best logp: {best_logp:.3f},",
            f"accepted: {accepted_bool}, p_accept: {num_accept / (i + start_index + 1) * 100:.1f}%",
        )

        if isinstance(sampler, samplers.DelayedRejectionAdaptiveMetropolis):
            proposal_cov = sampler.cov_chol
        else:
            proposal_cov = None

        if (i == max_samples) or (i % args.output_interval == 0):
            analysis.analyze_mcmc(
                root_dir.parent,
                os.path.basename(args.config_file),
                args.datasets,
                corner=True,
                bands=True,
                proposal_cov=proposal_cov,
            )

        if i >= max_samples:
            break

        update_opts(root_dir, i + start_index + 1)
