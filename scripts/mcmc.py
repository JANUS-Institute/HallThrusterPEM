import argparse
import os
import pickle
import shutil
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeAlias

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


@dataclass
class ExecutionOptions:
    executor: Type
    max_workers: int
    directory: Path
    fidelity: str | tuple | dict = (0, 0)


def load_system_and_opts(args: Namespace) -> tuple[System, ExecutionOptions]:
    config = args.config_file
    system = YamlLoader.load(config)
    system.root_dir = Path(config).parent
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


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    system, opts = load_system_and_opts(args)
    base = load_nominal_inputs(system)

    # Load data
    data = hallmd.data.load(spt100.macdonald2019() + spt100.diamant2014())
    operating_conditions = list(data.keys())

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys_forward[param.casefold()]
        inputs_unnorm = np.array([getattr(cond, long_name) for cond in data.keys()])
        base[param] = system.inputs()[param].normalize(inputs_unnorm)

    # Choose calibration params
    names = (p.name for p in system.inputs())
    params_to_calibrate = [p for _, p in sorted(zip(names, system.inputs())) if p.category == "calibration"]

    print(f"Number of operating conditions: {len(operating_conditions)}")

    init_sample = np.array([base[p] for p in params_to_calibrate])
    init_cov = np.diag(np.ones(len(params_to_calibrate)) * 0.5)

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

    max_samples: int = 250
    best_sample = init_sample
    best_logp = init_logp
    num_accept = 1
    output_interval = 10

    start_index = 1
    update_opts(root_dir, start_index)

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

        if (i == max_samples) or (i % output_interval == 0):
            analysis.analyze_mcmc(root_dir.parent, os.path.basename(args.config_file))

        if i >= max_samples:
            break

        update_opts(root_dir, i + start_index + 1)
