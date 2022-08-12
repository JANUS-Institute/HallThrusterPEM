# Module for plume models

import numpy as np
from scipy.special import erfi
import logging
import sys

Q_E = 1.602176634e-19   # Fundamental charge (C)
logger = logging.getLogger(__name__)
sys.path.append('..')

from utils import data_load, data_write


def current_density_model(plume_input='plume_input.json', thruster_input=None, write_output=True):
    logger.info('Running plume model')
    # Load plume inputs
    inputs = data_load(plume_input)
    theta = []
    for param, value in inputs['parameters'].items():
        theta.append(value)
    P_B = inputs['design']['PB_norm']
    I_B0 = inputs['other']['I_B0']
    r = np.array(inputs['other']['r'])
    alpha = np.array([alpha_val * np.pi/180 for alpha_val in inputs['other']['alpha']])
    sigma_cex = inputs['other']['sigma_cex']

    # Load inputs from thruster model
    if thruster_input:
        thruster_data = data_load(thruster_input)
        I_B0 = 0
        # Sum ion current density over all ion charge states at thruster exit
        for param, grid_sol in thruster_data[0].items():
            if 'niui' in param:
                charge_num = int(param.split('_')[1])
                I_B0 += Q_E * charge_num * grid_sol[-1]

    # Compute model prediction
    n = theta[4] * P_B + theta[5]
    alpha1 = theta[1] * (theta[2] * P_B + theta[3])
    alpha2 = (theta[2] * P_B + theta[3])

    A1 = (1 - theta[0]) / ((np.pi ** (3 / 2)) / 2 * alpha1 * np.exp(-(alpha1 / 2)**2) * (
                2 * erfi(alpha1 / 2) + erfi((np.pi * 1j - (alpha1 ** 2)) / (2 * alpha1)) - erfi(
            (np.pi * 1j + (alpha1 ** 2)) / (2 * alpha1))))
    A2 = theta[0] / ((np.pi ** (3 / 2)) / 2 * alpha2 * np.exp(-(alpha2 / 2)**2) * (
                2 * erfi(alpha2 / 2) + erfi((np.pi * 1j - (alpha2 ** 2)) / (2 * alpha2)) - erfi(
            (np.pi * 1j + (alpha2 ** 2)) / (2 * alpha2))))

    I_B = I_B0 * np.exp(-r*n*sigma_cex)
    j_beam = (I_B / r ** 2) * A1 * np.exp(-(alpha / alpha1) ** 2)
    j_scat = (I_B / r ** 2) * A2 * np.exp(-(alpha / alpha2) ** 2)
    j_cex = I_B0 * (1 - np.exp(-r*n*sigma_cex)) / (2 * np.pi * r ** 2)
    j = j_beam + j_scat + j_cex

    if np.any(abs(j.imag) > 0):
        logger.warning('Predicted beam current has imaginary component')

    if write_output:
        output_data = {"j_ion": list(j.real)}
        data_write(output_data, 'plume_output.json')

    return j.real
