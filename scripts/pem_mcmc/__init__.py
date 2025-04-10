import sys

import numpy as np
from amisc import Component, System, Variable

from pem_mcmc.analysis import analyze
from pem_mcmc.io import append_sample_row, read_output_file
from pem_mcmc.metrics import likelihood_and_distances, log_posterior
from pem_mcmc.options import ExecutionOptions, load_system_and_opts
from pem_mcmc.samplers import DRAMSampler, PreviousRunSampler, PriorSampler
from pem_mcmc.types import Value

__all__ = [
    "DRAMSampler",
    "PreviousRunSampler",
    "PriorSampler",
    "get_nominal_inputs",
    "load_system_and_opts",
    "append_sample_row",
    "load_calibration_variables",
    "analyze",
    "likelihood_and_distances",
    "log_posterior",
    "read_output_file",
    "ExecutionOptions",
]


def get_nominal_inputs(system: System) -> dict[str, Value]:
    """Create a dict mapping system inputs to their nominal values"""
    inputs: dict[str, Value] = {}
    for input in system.inputs():
        value = input.normalize(input.get_nominal())
        inputs[input.name] = value

    return inputs


def _print_amisc_output(output, file=sys.stdout, max_elems=10):
    print("{", file=file)
    indent = "   "

    for key, item in output.items():
        print(f"{indent}{key}: ", file=file, end="")

        if isinstance(item, np.ndarray):
            _item = np.array(item.squeeze())
            print(f"<ndarray of shape {_item.shape}>")
            with np.printoptions(precision=3, suppress=True, threshold=max_elems):
                print(f"array({_item}, shape={_item.shape})")
        else:
            print(f"{item},", file=file)

    print("}", file=file)


def _get_component_variables_by_category(
    component: Component, category: str, sort: str | None = None
) -> list[Variable]:
    """Extract variables of a given `category` from the provided `component`.
    Optionally sort them based on either their 'name' or 'tex' representations"""

    params = [p for p in component.inputs if p.category == category]
    if sort is None:
        return params

    names = [getattr(p, sort).casefold() for p in params]
    sorted_params = [p for _, p in sorted(zip(names, params))]
    return sorted_params


def load_calibration_variables(system: System, sort: str | None = None) -> list[Variable]:
    cathode_params = _get_component_variables_by_category(system['Cathode'], 'calibration', sort=sort)
    plume_params = _get_component_variables_by_category(system['Plume'], 'calibration', sort=sort)
    thruster_params = _get_component_variables_by_category(system['Thruster'], 'calibration', sort=sort)

    # assemble params_to_calibrate, has to be done in this way to avoid duplicate Te in cathode and thruster
    calibration_vars: list[Variable] = cathode_params
    calibration_vars += [p for p in thruster_params if p not in calibration_vars]
    calibration_vars += [p for p in plume_params if p not in calibration_vars]

    return calibration_vars
