import argparse
import os
import pickle
import shutil
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeAlias

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
    default=10,
    help="How frequently plots are generated",
)

parser.add_argument(
    "--ncharge",
    type=int,
    default=1,
    help="Number of ion charge states to include in simulation. Must be between 1 and 3.",
)

parser.add_argument("--output_dir", type=str, default=None, help="Directory into which output files are written")

parser.add_argument("--init_sample", type=str, default=None, help="CSV file containing initial sample")

parser.add_argument("--init_cov", type=str, default=None, help="CSV file containing initial covariance matrix")


@dataclass
class ExecutionOptions:
    executor: Type
    max_workers: int
    directory: Path
    fidelity: str | tuple | dict = (0, 0)
    use_plume: bool = True
    use_cathode: bool = True


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
        raise (e)


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
) -> np.float64:
    result = _run_model(params, system, list(data.keys()), base_params, opts)

    L = np.float64(0.0)
    for opcond, _data in data.items():
        L += ThrusterData.log_likelihood(_data, result[opcond])

    # geometric average of likelihood over operating conditions
    L /= len(list(data.keys()))

    # if somehow nan or otherwise non-finite, return -inf
    if not np.isfinite(L):
        return np.float64(-np.inf)

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
    if not np.isfinite(prior):
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


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    system, opts = load_system_and_opts(args)
    base = get_nominal_inputs(system)

    # Determine thruster and load data
    thruster_name = system['Thruster'].model_kwargs['thruster']
    thruster = hallmd.data.thrusters[thruster_name]
    datasets = thruster.datasets_from_names(args.datasets)
    data = hallmd.data.load(datasets)
    operating_conditions = list(data.keys())

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys_forward[param.casefold()]
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

    logpdf = lambda x: log_posterior(dict(zip(params_to_calibrate, x)), data, system, base, opts)
    root_dir = opts.directory
    logfile = root_dir / "mcmc.csv"

    # Prepare output file
    delimiter = ","
    header = delimiter.join(["id"] + [p.name for p in params_to_calibrate] + ["log_posterior"] + ["accepted"])
    with open(logfile, "w") as fd:
        print(header, file=fd)

    # Generate initial logpdf
    update_opts(root_dir, 0)
    init_logp = logpdf(init_sample)
    print(f"{init_logp}")

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

    max_samples: int = args.max_samples
    best_sample = init_sample
    best_logp = init_logp
    num_accept = 1

    start_index = 1
    update_opts(root_dir, start_index)

    # MCMC main loop
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

        if (i == max_samples) or (i % args.output_interval == 0):
            analysis.analyze_mcmc(
                root_dir.parent,
                os.path.basename(args.config_file),
                args.datasets,
                corner=True,
                bands=True,
                proposal_cov=sampler.cov_chol,
            )

        if i >= max_samples:
            break

        update_opts(root_dir, i + start_index + 1)
