"""Module for full multidisciplinary Hall thruster PEM"""

import numpy as np
import sys
import logging
import copy
from pathlib import Path
import pickle

sys.path.append('..')

# Custom imports
from models.cc import cathode_coupling_model_feedforward, cc_pem
from models.thruster import hall_thruster_jl_model, thruster_pem
from models.plume import jion_modified, plume_pem
from utils import UniformRV, LogUniformRV, LogNormalRV, NormalRV
from surrogates.system import SystemSurrogate


def feedforward_pem(model_inputs, jl=None):
    """Run a single vcc-thruster-plume model with dictionary inputs"""
    assert len(model_inputs) == 3

    # Allocate space for return dictionary
    pem_result = {'pem_version': 'feedforward', 'cc': {}, 'thruster': {}, 'plume': {}}

    # Run cathode-coupling model
    cc_input = model_inputs[0]
    cc_output = cathode_coupling_model_feedforward(cc_input)
    pem_result['cc']['input'] = cc_input
    pem_result['cc']['output'] = cc_output

    # Run Hallthruster.jl model
    thruster_input = model_inputs[1]
    pem_result['thruster']['input'] = copy.deepcopy(thruster_input)
    thruster_input.update(cc_output)
    thruster_output = hall_thruster_jl_model(thruster_input, jl=jl)
    pem_result['thruster']['output'] = thruster_output

    # # Run plume model
    plume_input = model_inputs[2]
    pem_result['plume']['input'] = copy.deepcopy(plume_input)
    plume_input.update(thruster_output)
    plume_output = jion_modified(plume_input)
    pem_result['plume']['output'] = plume_output

    return pem_result


def pem_system(root_dir=None, executor=None, init=True):
    """Return a SystemSurrogate object for the feedforward PEM system"""
    # Pressure is in Torr, mass flow rate in kg/s, and everything else base SI except where specified otherwise
    exo_vars = [UniformRV(-8, -3, 'PB'), UniformRV(200, 400, 'Va'), UniformRV(3, 7, 'mdot_a'),
                UniformRV(1, 5, 'T_ec'), UniformRV(0, 60, 'V_vac'), UniformRV(1, 10, 'P*'),
                UniformRV(1, 10, 'PT'), UniformRV(100, 500, 'u_n'), UniformRV(0.1, 0.3, 'c_w'),
                UniformRV(1, 20, 'l_t'), UniformRV(-3, -1, 'vAN1'), UniformRV(2, 100, 'vAN2'),
                UniformRV(0.07, 0.09, 'l_c'), UniformRV(800, 1200, 'Ti'), UniformRV(280, 320, 'Tn'),
                UniformRV(280, 320, 'Tb'), UniformRV(0, 1, 'c0'), UniformRV(0.1, 0.9, 'c1'), UniformRV(-15, 15, 'c2'),
                UniformRV(0, np.pi/2, 'c3'), UniformRV(18, 22, 'c4'), UniformRV(14, 18, 'c5'),
                UniformRV(51, 58, 'sigma_cex'), UniformRV(0.5, 1.5, 'r_m')]
    coupling_vars = [UniformRV(0, 60, 'V_cc'), UniformRV(0, 10, 'I_B0'), UniformRV(0, 0.2, 'T'),
                     UniformRV(0, 1, 'eta_v'), UniformRV(0, 1, 'eta_c'), UniformRV(0, 1, 'eta_m'),
                     UniformRV(14000, 22000, 'ui_avg')]

    # Get number of reconstruction coefficients
    with open(Path(__file__).parent / 'thruster_svd.pkl', 'rb') as fd, \
            open(Path(__file__).parent / 'plume_svd.pkl', 'rb') as fd2:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        d2 = pickle.load(fd2)
        r2 = d2['vtr'].shape[0]
        coupling_vars.extend([UniformRV(-20, 20, f'uion{i}') for i in range(r1)])
        coupling_vars.append(UniformRV(0, np.pi/2, 'theta_d'))  # Divergence angle
        coupling_vars.extend([UniformRV(-20, 20, f'jion{i}') for i in range(r2)])

    cathode_exo = [0, 1, 3, 4, 5, 6]
    thruster_exo = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plume_exo = [0, 16, 17, 18, 19, 20, 21, 22, 23]
    # Models must be specified at the global scope for pickling
    cathode = {'name': 'Cathode', 'model': cc_pem, 'truth_alpha': (), 'exo_in': cathode_exo, 'local_in': {},
               'global_out': [0], 'max_alpha': (), 'max_beta': (3,)*len(cathode_exo), 'type': 'analytical',
               'model_args': (), 'model_kwargs': {}}
    thruster = {'name': 'Thruster', 'model': thruster_pem, 'truth_alpha': (3, 2), 'max_alpha': (3, 2),
                'exo_in': thruster_exo, 'local_in': {'Cathode': [0]}, 'global_out': list(np.arange(1, 6+r1+1)),
                'type': 'lagrange', 'max_beta': (2,) * (len(thruster_exo) + 1), 'save_output': True,
                'model_args': (), 'model_kwargs': {'n_jobs': -1, 'compress': True}}
    plume = {'name': 'Plume', 'model': plume_pem, 'truth_alpha': (), 'exo_in': plume_exo, 'max_alpha': (),
             'local_in': {'Thruster': [0]}, 'global_out': list(np.arange(6+r1+1, len(coupling_vars))),
             'type': 'analytical', 'max_beta': (3,)*(len(plume_exo)+1), 'model_args': (),
             'model_kwargs': {'compress': True}}
    sys = SystemSurrogate([cathode, thruster, plume], exo_vars, coupling_vars, executor=executor, init_surr=init,
                          suppress_stdout=True)

    return sys
