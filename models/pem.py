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
    exo_vars = [UniformRV(-8, -3, disp='PB', long_label='Background pressure magnitude (Torr)'),
                UniformRV(200, 400, disp='Va', long_label='Anode voltage (V)'),
                UniformRV(3, 7, disp='mdot_a', long_label='Anode mass flow rate (mg/s)'),
                UniformRV(1, 5, disp='T_ec', long_label='Electron temperature at cathode (eV)'),
                UniformRV(0, 60, disp='V_vac', long_label='Vacuum coupling voltage (V)'),
                UniformRV(1, 10, disp='P*', long_label='Turning point pressure (Torr)'),
                UniformRV(1, 10, disp='PT', long_label='Thruster to facility plasma density ratio (Torr)'),
                UniformRV(100, 500, disp='u_n', long_label='Neutral velocity (m/s)'),
                UniformRV(0.1, 0.3, disp='c_w', long_label='Wall sheath loss coefficient (-)'),
                UniformRV(1, 20, disp='l_t', long_label='Transition length (mm)'),
                UniformRV(-3, -1, disp='vAN1', long_label='Anomalous coefficient 1 (-)'),
                UniformRV(2, 100, disp='vAN2', long_label='Anomalous coefficient 2 (-)'),
                UniformRV(0.07, 0.09, disp='l_c', long_label='Axial cathode location (m)'),
                UniformRV(800, 1200, disp='Ti', long_label='Ion temperature (K)'),
                UniformRV(280, 320, disp='Tn', long_label='Neutral temperature (K)'),
                UniformRV(280, 320, disp='Tb', long_label='Background temperature (K)'),
                UniformRV(0, 1, disp='c0', long_label='Plume fit coefficient 0 (-)'),
                UniformRV(0.1, 0.9, disp='c1', long_label='Plume fit coefficient 1 (-)'),
                UniformRV(-15, 15, disp='c2', long_label='Plume fit coefficient 2 (-)'),
                UniformRV(0, np.pi/2, disp='c3', long_label='Plume fit coefficient 3 (-)'),
                UniformRV(18, 22, disp='c4', long_label='Plume fit coefficient 4 (-)'),
                UniformRV(14, 18, disp='c5', long_label='Plume fit coefficient 5 (-)'),
                UniformRV(51, 58, disp='sigma_cex', long_label='Charge exchange collision cross-section (Ang^2)'),
                UniformRV(0.5, 1.5, disp='r_m', long_label='Distance from thruster exit plane (m)')]
    coupling_vars = [UniformRV(0, 60, disp='V_cc', long_label='Cathode coupling voltage (V)'),
                     UniformRV(0, 10, disp='I_B0', long_label='Beam current at channel exit (A)'),
                     UniformRV(0, 0.2, disp='T', long_label='Thrust (N)'),
                     UniformRV(0, 1, disp='eta_v', long_label='Voltage efficiency (-)'),
                     UniformRV(0, 1, disp='eta_c', long_label='Current efficiency (-)'),
                     UniformRV(0, 1, disp='eta_m', long_label='Mass utilization efficiency (-)'),
                     UniformRV(14000, 22000, disp='ui_avg', long_label='Average ion exit velocity (m/s)')]

    # Get number of reconstruction coefficients
    with open(Path(__file__).parent / 'thruster_svd.pkl', 'rb') as fd, \
            open(Path(__file__).parent / 'plume_svd.pkl', 'rb') as fd2:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        d2 = pickle.load(fd2)
        r2 = d2['vtr'].shape[0]
        coupling_vars.extend([UniformRV(-20, 20, disp=f'uion{i}') for i in range(r1)])
        coupling_vars.append(UniformRV(0, np.pi/2, disp='theta_d', long_label='Divergence angle (rad)'))  # Div angle
        coupling_vars.extend([UniformRV(-20, 20, disp=f'jion{i}') for i in range(r2)])

    cathode_exo = [0, 1, 3, 4, 5, 6]
    thruster_exo = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plume_exo = [0, 16, 17, 18, 19, 20, 21, 22, 23]
    # Models must be specified at the global scope for pickling
    cathode = {'name': 'Cathode', 'model': cc_pem, 'truth_alpha': (), 'exo_in': cathode_exo, 'local_in': {},
               'global_out': [0], 'max_alpha': (), 'max_beta': (3,)*len(cathode_exo), 'type': 'analytical',
               'model_args': (), 'model_kwargs': {}}
    thruster = {'name': 'Thruster', 'model': thruster_pem, 'truth_alpha': (2, 2), 'max_alpha': (2, 2),
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
