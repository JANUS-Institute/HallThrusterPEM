# Module for plume models

import numpy as np
from scipy.special import erfi
import logging
import sys
import math
import scipy.integrate

sys.path.append('..')
Q_E = 1.602176634e-19   # Fundamental charge (C)
logger = logging.getLogger(__name__)

from utils import ModelRunException


def current_density_model(plume_input, N=50):
    # logger.info('Running plume model')

    # Load plume inputs
    theta = [plume_input['c0'], plume_input['c1'], plume_input['c2'], plume_input['c3'], plume_input['c4'],
             plume_input['c5']]
    P_B = plume_input['background_pressure_Torr'] / plume_input['p_ref']
    I_B0 = plume_input['I_B0']
    sigma_cex = plume_input['sigma_cex']

    # Format the prediction locations
    if len(plume_input['r_m']) == 1 == len(plume_input['alpha_deg']):
        # Predict at one location
        r = np.atleast_1d(plume_input['r_m'])
        alpha = np.atleast_1d(plume_input['alpha_deg'])
    elif len(plume_input['r_m']) == 1 and len(plume_input['alpha_deg']) == 2:
        # Predict over a sweep of angles
        alpha = np.linspace(*plume_input['alpha_deg'], N)
        r = np.ones(N) * plume_input['r_m'][0]
    elif len(plume_input['r_m']) == 2 and len(plume_input['alpha_deg']) == 1:
        # Predict over a sweep of radii
        r = np.linspace(*plume_input['r_m'], N)
        alpha = np.ones(N) * plume_input['alpha_deg'][0]
    elif len(plume_input['r_m']) == 2 == len(plume_input['alpha_deg']):
        # Predict over a grid of r, alpha
        loc = [np.linspace(*plume_input['r_m'], N), np.linspace(*plume_input['alpha_deg'], N)]
        pt_grids = np.meshgrid(*loc)
        x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
        r = x_loc[:, 0]
        alpha = x_loc[:, 1]
    else:
        raise Exception('(r, alpha) locations ill-specified')

    # Append cathode prediction location
    r = np.append(r, plume_input['r_cathode'])
    alpha = np.append(alpha, plume_input['alpha_cathode']) * np.pi / 180

    # Compute model prediction
    with np.errstate(invalid='raise', divide='raise'):
        try:
            n = max(0, theta[4] * P_B + theta[5])
            alpha1 = theta[1] * (theta[2] * P_B + theta[3])
            alpha2 = (theta[2] * P_B + theta[3])

            # Threshold values [0, pi/2]
            tol = 1e-20
            alpha1 = min(math.pi/2, max(tol, alpha1))
            alpha2 = min(math.pi/2, max(tol, alpha2))

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

            # Return quantities of interest
            j_cathode = j[-1]
            j_ion = j[:-1]
            r = r[:-1]
            alpha = alpha[:-1]

            # Calculate divergence angle from https://aip.scitation.org/doi/10.1063/5.0066849
            # Assumes alpha = [-90, 90]
            start_idx = np.argmin(np.abs(alpha-0))  # Start at channel centerline
            num_int = j_ion * np.cos(alpha) * np.sin(alpha)
            den_int = j_ion * np.cos(alpha)
            cos_div = scipy.integrate.simps(num_int[start_idx:], alpha[start_idx:]) / \
                      scipy.integrate.simps(den_int[start_idx:], alpha[start_idx:])

        except Exception as e:
            raise ModelRunException(f"Exception in plume model: {e}")

        else:
            return {'r': list(r), 'alpha': list(alpha), 'ion_current_density': list(j_ion.real),
                    'cathode_current_density': float(j_cathode.real), 'divergence_angle': np.arccos(cos_div),
                    'divergence_efficiency': cos_div**2}
