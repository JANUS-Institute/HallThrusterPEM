"""Test the thruster models."""
import copy
import json

import numpy as np
import matplotlib.pyplot as plt

from hallmd.models.thruster import hallthruster_jl, _convert_to_julia, _convert_to_pem, PEM_TO_JULIA


def test_julia_conversion():
    """Test that we can set arbitrary values in a HallThruster config struct from corresponding PEM values."""
    pem = {'V_a': 250, 'trough_loc': 0.1, 'T': 2, 'new_var': 0.5}
    julia = {'config': {'discharge_voltage': 100, 'anom_model': {'model': {'trough_location': 0.2}}}}
    pem_to_julia = copy.deepcopy(PEM_TO_JULIA)
    pem_to_julia['new_var'] = ['new', 1, 'expanded_variable_name']
    pem_to_julia['new_output'] = ['outputs', 'time_resolved', 'long_output_name']

    _convert_to_julia(pem, julia, pem_to_julia)

    assert julia['config']['discharge_voltage'] == 250
    assert julia['config']['anom_model']['model']['trough_location'] == 0.1
    assert julia['outputs']['average']['thrust'] == 2
    assert isinstance(julia['new'], list)
    assert len(julia['new']) == 2
    assert julia['new'][0] == {}
    assert julia['new'][1]['expanded_variable_name'] == 0.5

    # Test the reverse conversion
    julia['outputs'].update({'time_resolved': {'long_output_name': 0.5}})
    pem_convert = _convert_to_pem(julia, pem_to_julia)
    assert pem_convert['T'] == 2
    assert pem_convert['new_output'] == 0.5


def test_sim_hallthruster_jl(tmp_path, plots=False):
    """Simulate a fake HallThruster.jl model to test the Python wrapper function."""
    thruster_inputs = {'V_a': 250, 'V_cc': 25, 'mdot_a': 3.5e-6}
    config = {'anom_model': {'type': 'PressureShifted', 'model': {
        'type': 'TwoZoneBohm',
        'c1': 0.008,
        'c2': 0.08
    }}, 'domain': [0, 0.08]}
    simulation = {'ncells': 100, 'duration': 1e-3, 'dt': 1e-9}
    postprocess = {'average_start_time': 0.5e-3}

    # Run the simulation
    outputs = hallthruster_jl(thruster_inputs, config=config, simulation=simulation, postprocess=postprocess,
                              julia_script='sim_hallthruster.jl', output_path=tmp_path)

    if plots:
        with open(tmp_path / outputs['output_path'], 'r') as fd:
            data = json.load(fd)
            z = np.atleast_1d(data['outputs']['average']['z'])
            u_ion = np.atleast_1d(outputs['u_ion'])

        fig, ax = plt.subplots()
        ax.plot(z, u_ion, '-k')
        ax.set_xlabel('Axial distance from anode (m)')
        ax.set_ylabel('Ion velocity (m/s)')
        ax.grid()

        plt.show()


def test_run_hallthruster_jl(plots=False):
    """Test actually calling HallThruster.jl using the Python wrapper function."""
    # TODO: Implement test
