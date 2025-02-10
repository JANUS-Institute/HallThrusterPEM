"""Test the thruster models."""

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hallmd.models.thruster import (
    HALLTHRUSTER_VERSION_TEST,
    PEM_TO_JULIA,
    _convert_to_julia,
    _convert_to_pem,
    hallthruster_jl,
)

SHOW_PLOTS = False


def test_julia_conversion():
    """Test that we can set arbitrary values in a HallThruster config struct from corresponding PEM values."""
    pem = {"V_a": 250, "anom_center": 0.1, "T": 2, "new_var": 0.5}
    julia = {"config": {"discharge_voltage": 100, "anom_model": {"model": {"center": 0.2}}}}
    pem_to_julia = copy.deepcopy(PEM_TO_JULIA)
    pem_to_julia["new_var"] = ["new", 1, "expanded_variable_name"]
    pem_to_julia["new_output"] = ["output", "time_resolved", "long_output_name"]

    _convert_to_julia(pem, julia, pem_to_julia)

    assert julia["config"]["discharge_voltage"] == 250
    assert julia["config"]["anom_model"]["model"]["center"] == 0.1
    assert julia["output"]["average"]["thrust"] == 2
    assert isinstance(julia["new"], list)
    assert len(julia["new"]) == 2
    assert julia["new"][0] == {}
    assert julia["new"][1]["expanded_variable_name"] == 0.5

    # Test the reverse conversion
    julia["output"].update({"time_resolved": {"long_output_name": 0.5}})
    pem_convert = _convert_to_pem(julia, pem_to_julia)
    assert pem_convert["T"] == 2
    assert pem_convert["new_output"] == 0.5


def test_sim_hallthruster_jl(tmp_path, plots=SHOW_PLOTS, git_ref=HALLTHRUSTER_VERSION_TEST):
    """Simulate a fake HallThruster.jl model to test the Python wrapper function."""
    thruster_inputs = {"V_a": 250, "V_cc": 25, "mdot_a": 3.5e-6}
    config = {
        "anom_model": {"type": "LogisticPressureShift", "model": {"type": "TwoZoneBohm", "c1": 0.008, "c2": 0.08}},
        "domain": [0, 0.08],
    }
    simulation = {"grid": {"type": "EvenGrid", "num_cells": 100}, "duration": 1e-3, "dt": 1e-9}
    postprocess = {"average_start_time": 0.5e-3}

    # Run the simulation
    outputs = hallthruster_jl(
        thruster_inputs,
        config=config,
        simulation=simulation,
        postprocess=postprocess,
        julia_script=(Path(__file__).parent / "sim_hallthruster.jl").resolve(),
        output_path=tmp_path,
        version=git_ref,
    )

    for key in ["T", "I_B0", "I_d", "u_ion", "u_ion_coords"]:
        assert key in outputs

    with open(tmp_path / outputs["output_path"], "r") as fd:
        data = json.load(fd)

        for key in ["thrust", "ion_current", "discharge_current", "mass_eff", "voltage_eff", "current_eff"]:
            assert key in data["output"]["average"]

    if plots:
        with open(tmp_path / outputs["output_path"], "r") as fd:
            data = json.load(fd)
            z = np.atleast_1d(data["output"]["average"]["z"])
            u_ion = np.atleast_1d(outputs["u_ion"])

        _, ax = plt.subplots()
        ax.plot(z, u_ion, "-k")
        ax.set_xlabel("Axial distance from anode (m)")
        ax.set_ylabel("Ion velocity (m/s)")
        ax.grid()

        plt.show()


def test_run_hallthruster_jl(tmp_path, plots=SHOW_PLOTS, git_ref=HALLTHRUSTER_VERSION_TEST):
    """Test actually calling HallThruster.jl using the Python wrapper function (with PEMv0 settings)."""
    num_cells = 200
    pem_inputs = {
        "V_a": 300,
        "mdot_a": 5.16e-6,
        "P_b": 3.5e-5,
        "V_cc": 32.5,
        "T_e": 1.33,
        "u_n": 141.2,
        "l_t": 1.88e-3,
        "a_1": 0.0068,
        "a_2": 14.6,
        "dz": 0.4,
        "z0": -0.031,
        "p0": 57e-6,
    }
    config = {
        "anom_model": {
            "type": "LogisticPressureShift",
            "model": {"type": "TwoZoneBohm", "c1": 0.00625, "c2": 0.0625},
            "dz": 0.2,
            "z0": -0.03,
            "pstar": 45e-6,
            "alpha": 14,
        },
        "domain": [0, 0.08],
        "propellant": "Xenon",
        "ion_wall_losses": True,
    }

    simulation = {
        "grid": {"type": "EvenGrid", "num_cells": num_cells},
        "duration": 2e-3,
        "adaptive": True,
        "verbose": False,
    }
    postprocess = {"average_start_time": 1e-3}
    model_fidelity = (int(num_cells / 50) - 2, 2)
    thruster = "SPT-100"

    outputs = hallthruster_jl(
        pem_inputs,
        config=config,
        simulation=simulation,
        postprocess=postprocess,
        model_fidelity=model_fidelity,
        thruster=thruster,
        version=git_ref,
        output_path=tmp_path,
    )
    pem_inputs["z0"] = -0.15
    outputs_shift = hallthruster_jl(
        pem_inputs,
        config=config,
        simulation=simulation,
        postprocess=postprocess,
        model_fidelity=model_fidelity,
        thruster=thruster,
        version=git_ref,
        output_path=tmp_path,
    )
    print(f"Cost: {outputs['model_cost']} s")

    for key in ["T", "I_B0", "I_d", "u_ion", "u_ion_coords"]:
        assert key in outputs

    assert len(outputs["u_ion_coords"]) == num_cells + 2
    assert len(outputs["u_ion"]) == num_cells + 2
    assert 0 < outputs["T"] < 0.2
    assert 0 < outputs["I_B0"] < 10
    assert 0 < outputs["I_d"] < 10

    with open(tmp_path / outputs["output_path"], "r") as fd:
        data = json.load(fd)
        nu_anom = data["output"]["average"]["nu_anom"]
        B = data["output"]["average"]["B"]
        Tev = data["output"]["average"]["Tev"]

        for key in [
            "thrust",
            "ion_current",
            "discharge_current",
            "mass_eff",
            "voltage_eff",
            "current_eff",
            "anode_eff",
            "ui",
        ]:
            assert key in data["output"]["average"]

    if plots:
        _, ax = plt.subplots(1, 3, figsize=(12, 4), layout="tight")
        grid = outputs["u_ion_coords"]
        ax[0].plot(grid, outputs["u_ion"], "-k", label="$z_0 = 0.03$")
        ax[0].plot(grid, outputs_shift["u_ion"], "--r", label="$z_0 = 0.15$")
        ax[0].set_xlabel("Axial distance from anode (m)")
        ax[0].set_ylabel("Ion velocity (m/s)")
        ax[0].legend()
        ax[0].grid()

        with open(tmp_path / outputs_shift["output_path"], "r") as fd:
            data = json.load(fd)
            anom_shift = data["output"]["average"]["nu_anom"]

        ax[1].plot(grid, nu_anom, "-k", label="$z_0 = 0.03$")
        ax[1].plot(grid, anom_shift, "--r", label="$z_0 = 0.15$")
        ax[1].set_xlabel("Axial distance from anode (m)")
        ax[1].set_ylabel("Anomalous collision frequency (Hz)")
        ax[1].set_yscale("log")
        ax[1].legend()
        ax[1].grid()

        ax[2].plot(grid, np.array(B) / np.max(B), "-b", label="Magnetic field")
        ax[2].plot(grid, np.array(Tev) / np.max(Tev), "-r", label="Electron temperature")
        ax[2].set_xlabel("Axial distance from anode (m)")
        ax[2].set_ylabel("Normalized values")
        ax[2].legend()
        ax[2].grid()

        plt.show()
