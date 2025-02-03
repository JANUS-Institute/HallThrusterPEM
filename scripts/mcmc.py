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
from amisc import System, YamlLoader
from MCMCIterators import samplers

import hallmd.data
import hallmd.data.spt100 as spt100
from hallmd.data import Array, OperatingCondition

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
    help="A list of datasets to use, pick from [diamant2014, macdonald2019, sankovic1993]",
)

parser.add_argument(
    "--output_interval",
    type=int,
    default=10,
    help="How frequently plots are generated",
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
        fidelity=(0, 0),
    )

    os.mkdir(opts.directory)

    return system, opts


def load_nominal_inputs(system: System) -> dict[str, Value]:
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

    # Run model
    with opts.executor(max_workers=opts.max_workers) as exec:
        outputs = system.predict(
            sample_dict,
            use_model=opts.fidelity,
            model_dir=opts.directory,
            executor=exec,
            verbose=False,
        )

    # Assemble output dict from operating conditions -> results
    output_thrusterdata = hallmd.data.pem_to_thrusterdata(operating_conditions, outputs)

    # Write outputs to file
    with open(opts.directory / "pemv1.pkl", "wb") as fd:
        pickle.dump({"input": sample_dict, "output": output_thrusterdata}, fd)

    return output_thrusterdata


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
    data: dict[OperatingCondition, hallmd.data.ThrusterData],
    system: System,
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> np.float64:
    result = _run_model(params, system, list(data.keys()), base_params, opts)

    L = np.float64(0.0)
    for opcond, _data in data.items():
        L += hallmd.data.log_likelihood(_data, result[opcond])

    return L


def log_posterior(
    params: dict[str, Value],
    data: dict[OperatingCondition, hallmd.data.ThrusterData],
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


def write_row(logfile, sample_index, sample, logp, accepted_bool):
    id_str = f"{sample_index:06d}"
    with open(logfile, "a") as fd:
        row = [id_str] + [f"{s}" for s in sample] + [f"{logp}", f"{accepted_bool}"]
        print(delimiter.join(row), file=fd)


def read_csv(file):
    with open(file, 'r') as fd:
        header = fd.readline().rstrip()
        if header.startswith("#"):
            header = header[1:].lstrip()

    col_names = header.split(',')
    table_data = np.genfromtxt(file, skip_header=1).T
    return {col_name: column for (col_name, column) in zip(col_names, table_data)}


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    system, opts = load_system_and_opts(args)
    base = load_nominal_inputs(system)

    # Load data
    data = hallmd.data.load(spt100.datasets_from_names(args.datasets))
    operating_conditions = list(data.keys())

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys_forward[param.casefold()]
        inputs_unnorm = np.array([getattr(cond, long_name) for cond in data.keys()])
        base[param] = system.inputs()[param].normalize(inputs_unnorm)

    # Choose calibration params
    if "diamant2014" in args.datasets:
        names = (p.name for p in system.inputs())
        params_to_calibrate = [p for _, p in sorted(zip(names, system.inputs())) if p.category == "calibration"]
    else:
        params_to_calibrate = [
            system.inputs()[p]
            for p in [
                "anom_min",
                "anom_max",
                "anom_width",
                "anom_center",
                "anom_shift_length",
                "c_w",
                "u_n",
            ]
        ]

    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Create initial sample
    index_map = {p.name: i for (i, p) in enumerate(params_to_calibrate)}
    init_sample = np.array([base[p] for p in params_to_calibrate])
    if args.init_sample is not None:
        var_dict = read_csv(args.init_sample)
        for k, v in var_dict.items():
            i = index_map[k]
            init_sample[i] = v

        print(params_to_calibrate)
        print(var_dict)
        print(init_sample)

    # Create initial covariance
    if args.init_cov is None:
        variances = np.ones(len(params_to_calibrate))

        for i, p in enumerate(params_to_calibrate):
            dist = system.inputs()[p].distribution
            if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                lb, ub = dist.dist_args
                std = (ub - lb) / 10
            elif isinstance(dist, distributions.Normal):
                std = dist.dist_args[1]
            elif isinstance(dist, distributions.LogNormal):
                std = dist.base ** dist.dist_args[1]
            else:
                raise ValueError(f"Invalid distribution {dist}")

            variances[i] = system.inputs()[p].normalize(std) ** 2

        init_cov = np.diag(variances)
    else:
        cov_dict = read_csv(args.init_cov)
        N = len(params_to_calibrate)
        init_cov = np.zeros((N, N))

        for i, (key1, column) in enumerate(cov_dict.items()):
            i1 = index_map[key1]
            for j, (var, key2) in enumerate(zip(column, cov_dict.keys())):
                i2 = index_map[key2]
                init_cov[i1, i2] = var

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

    # Normalize logpdf by initial value
    norm_const = abs(init_logp)
    logpdf = lambda x: log_posterior(dict(zip(params_to_calibrate, x)), data, system, base, opts) / norm_const
    init_logp = init_logp / norm_const

    write_row(logfile, 0, init_sample, init_logp, True)

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
        write_row(logfile, i + start_index, sample, logp, accepted_bool)

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
                root_dir.parent, os.path.basename(args.config_file), args.datasets, corner=True, bands=True
            )

        if i >= max_samples:
            break

        update_opts(root_dir, i + start_index + 1)
