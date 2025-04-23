import copy
import pickle
import sys
import traceback
from dataclasses import fields
from typing import Optional

import amisc.distribution
import numpy as np
from amisc import System

import hallmd.data
from hallmd.data import OperatingCondition, ThrusterData
from pem_mcmc.options import ExecutionOptions
from pem_mcmc.types import Array, Dataset, Inf, Value


def _gauss_logpdf_1D(x: np.float64 | float, mean: np.float64 | float, std: np.float64 | float) -> float:
    """
    Gaussian log-likelihood in 1D
    """
    return -0.5 * (np.log(2 * np.pi * std**2) + (x - mean) ** 2 / std**2)


def _relative_l2_norm(data: Array, observation: Array) -> float:
    """
    Compute the L2-normed distance between a data array and observation array,
    relative to the magnitude of the data array.
    """
    num = np.sum((data - observation) ** 2)
    denom = np.sum(data**2)
    return np.sqrt(num / denom)


def _get_qoi_single_opcond(data: ThrusterData, observation: ThrusterData, field) -> Optional[tuple[Array, Array]]:
    data_field = getattr(data, field)
    obs_field = getattr(observation, field)
    """
    For a specified field in the `ThrusterData` class, extract the value of that field.
    If the field is not present in both the data and observation, returns None.
    Otherwise, returns a tuple of the (data vals, obs. vals) where both are 1D numpy arrays of the same size.
    """

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


def _get_qoi_all_opconds(data: Dataset, observation: Dataset, field: str) -> Optional[tuple[Array, Array]]:
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


def _log_prior(system: System, params: dict[str, Value]) -> np.float64:
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


def likelihood_and_distances(data, result, std=0.05) -> tuple[float, dict[str, tuple[float, float]]]:
    component_stats: dict[str, tuple[float, float]] = {}

    L = 0.0
    for field in fields(ThrusterData):
        # Assemble a vector of all data points for a single operating condition
        fieldname = field.name
        out = _get_qoi_all_opconds(data, result, fieldname)
        if out is None:
            continue

        data_arr, obs_arr = out
        distance = _relative_l2_norm(data_arr, obs_arr)

        # Since the l2-norm is positive-definite, the distribution is a half-normal, so we
        # need to multiply the normal pdf by 2 (or add ln(2) to the log-pdf)
        likelihood = _gauss_logpdf_1D(distance, 0.0, std) + np.log(2)

        component_stats[fieldname] = (distance, likelihood)

        L += likelihood

    return L, component_stats


def _log_likelihood(
    params: dict[str, Value],
    data: Dataset,
    system: System,
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> np.float64 | float:
    result = _run_model(params, system, list(data.keys()), base_params, opts)

    if result is None:
        return -np.inf

    L, component_stats = likelihood_and_distances(data, result, opts.noise_std)

    field_names = [f.name for f in fields(ThrusterData)]

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


def _run_model(
    params: dict[str, Value],
    system: System,
    operating_conditions: list[OperatingCondition],
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> Dataset | None:
    sample_dict = copy.deepcopy(base_params)
    for key, val in params.items():
        sample_dict[key] = val

    # Add aleatoric variables if requested
    if opts.sample_aleatoric:
        sample_dict = _sample_aleatoric(sample_dict, system)

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

        # _print_amisc_output(outputs)

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

        return output_thrusterdata

    except Exception as e:
        print("Error detected. Failing input printed below")
        print(f"{sample_dict}")
        print(e)
        traceback.print_exc(file=sys.stdout)

        return None


def log_posterior(
    params: dict[str, Value],
    data: Dataset,
    system: System,
    base_params: dict[str, Value],
    opts: ExecutionOptions,
) -> np.float64:
    prior = _log_prior(system, params)
    if not np.isfinite(prior):
        return -Inf

    likelihood = _log_likelihood(params, data, system, base_params, opts)
    if not np.isfinite(likelihood):
        return -Inf

    return prior + likelihood


def _sample_aleatoric(sample_dict, system):
    # We sample from normal distributions, as these are more reflective of the experimental uncertainty than
    # the uniform distributions that amisc uses for Relative.
    aleatoric_vars = ["P_b", "V_a", "mdot_a"]
    for var in aleatoric_vars:
        nominal = sample_dict[var]
        input = system.inputs()[var]
        dist = input.distribution
        assert isinstance(dist, amisc.distribution.Relative)

        # we assume the error specified in the config file is 1 standard deviation
        percent = dist.dist_args[0] / 100

        for i in range(np.atleast_1d(nominal).size):
            nom = input.denormalize(nominal[i])
            sample = nom * (1 + percent * np.random.randn())
            nominal[i] = input.normalize(sample)

        sample_dict[var] = nominal
    return sample_dict
