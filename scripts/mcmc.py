import argparse
import os
import shutil
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeAlias

import mcmc_plotting
import numpy as np
from amisc import System, YamlLoader

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
    fidelity: str | tuple | dict = (0, 0)
    directory: str | Path | None = None


def load_system_and_opts(args: Namespace) -> tuple[System, ExecutionOptions]:
    config = args.config_file
    system = YamlLoader.load(config)
    system.root_dir = Path(config).parent
    system.set_logger(stdout=True)
    if Path(config).name not in os.listdir(system.root_dir):
        shutil.copy(config, system.root_dir)

    match args.executor.lower():
        case "thread":
            executor = ThreadPoolExecutor
        case "process":
            executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    opts = ExecutionOptions(executor, args.max_workers, (0, 0), system.root_dir)

    return system, opts


def load_nominal_inputs(system: System) -> dict[str, Value]:
    inputs: dict[str, Value] = {}
    for input in system.inputs():
        value = input.normalize(input.get_nominal())
        inputs[input.name] = value

    return inputs


def _get_operating_conditions(system: System, input_dict: dict[str, Value]) -> list[OperatingCondition]:
    short_names = [short for short, _ in hallmd.data.opcond_keys_forward.items()]
    inputs = system.inputs()

    opcond_values = [inputs[short].denormalize(input_dict[short]) for short in short_names]

    opconds = [OperatingCondition(*values) for values in zip(*opcond_values)]

    return opconds


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

    logp = lambda x: log_posterior(dict(zip(params_to_calibrate, x)), data, system, base, opts)

    init_outputs = _run_model(dict(zip(params_to_calibrate, init_sample)), system, operating_conditions, base, opts)

    dev_spt100 = mcmc_plotting.Device(L_ch=0.025)

    mcmc_plotting.plot_u_ion(init_outputs, data, dev_spt100, "init")
