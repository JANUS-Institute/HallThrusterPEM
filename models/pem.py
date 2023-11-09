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
from models.thruster import hallthruster_jl_model, thruster_pem
from models.plume import jion_modified, plume_pem
from utils import UniformRV, load_variables
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
    thruster_output = hallthruster_jl_model(thruster_input, jl=jl)
    pem_result['thruster']['output'] = thruster_output

    # # Run plume model
    plume_input = model_inputs[2]
    pem_result['plume']['input'] = copy.deepcopy(plume_input)
    plume_input.update(thruster_output)
    plume_output = jion_modified(plume_input)
    pem_result['plume']['output'] = plume_output

    return pem_result


def pem_system(root_dir=None, executor=None, init=True, hf_override=False):
    """Return a SystemSurrogate object for the feedforward PEM system"""
    exo_vars = load_variables(['PB', 'Va', 'mdot_a', 'T_ec', 'V_vac', 'P*', 'PT', 'u_n', 'l_t', 'vAN1', 'vAN2',
                               'delta_z', 'z0*', 'p_u', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'sigma_cex', 'r_m'])
    coupling_vars = load_variables(['V_cc', 'I_B0', 'T', 'eta_v', 'eta_c', 'eta_m', 'ui_avg', 'theta_d'])

    # Get number of reconstruction coefficients
    with open(Path(__file__).parent / 'data' / 'thruster_svd.pkl', 'rb') as fd, \
            open(Path(__file__).parent / 'data' / 'plume_svd.pkl', 'rb') as fd2:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        d2 = pickle.load(fd2)
        r2 = d2['vtr'].shape[0]
        coupling_vars.extend([UniformRV(-20, 20, id=f'uion{i}', tex=f"$\\tilde{{u}}_{{ion,{i}}}$",
                                        description=f'Ion velocity latent coefficient {i}',
                                        param_type='coupling') for i in range(r1)])
        coupling_vars.extend([UniformRV(-20, 20, id=f'jion{i}', tex=f"$\\tilde{{j}}_{{ion,{i}}}$",
                                        description=f'Current density latent coefficient {i}',
                                        param_type='coupling') for i in range(r2)])

    cathode_exo = ['PB', 'Va', 'T_ec', 'V_vac', 'P*', 'PT']
    thruster_exo = ['PB', 'Va', 'mdot_a', 'T_ec', 'u_n', 'l_t', 'vAN1', 'vAN2', 'delta_z', 'z0*', 'p_u']
    plume_exo = ['PB', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'sigma_cex', 'r_m']
    # Models must be specified at the global scope for pickling
    cathode = {'name': 'Cathode', 'model': cc_pem, 'truth_alpha': (), 'exo_in': cathode_exo, 'coupling_in': {},
               'coupling_out': ['V_cc'], 'max_alpha': (), 'max_beta': (3,)*len(cathode_exo), 'type': 'analytical',
               'model_args': (), 'model_kwargs': {}}
    thruster = {'name': 'Thruster', 'model': thruster_pem, 'truth_alpha': (2, 2), 'max_alpha': (2, 2),
                'exo_in': thruster_exo, 'coupling_in': ['V_cc'], 'coupling_out':
                    ['I_B0', 'T', 'eta_v', 'eta_c', 'eta_m', 'ui_avg'] + [f'uion{i}' for i in range(r1)],
                'type': 'lagrange', 'max_beta': (2,) * (len(thruster_exo) + 1), 'save_output': True,
                'model_args': (), 'model_kwargs': {'n_jobs': -1, 'compress': True, 'hf_override': hf_override}}
    plume = {'name': 'Plume', 'model': plume_pem, 'truth_alpha': (), 'exo_in': plume_exo, 'max_alpha': (),
             'coupling_in': ['I_B0'], 'coupling_out': ['theta_d'] + [f'jion{i}' for i in range(r2)],
             'type': 'analytical', 'max_beta': (3,)*(len(plume_exo)+1), 'model_args': (),
             'model_kwargs': {'compress': True}}
    surr = SystemSurrogate([cathode, thruster, plume], exo_vars, coupling_vars, executor=executor, init_surr=init,
                           stdout=False, root_dir=root_dir)

    return surr
