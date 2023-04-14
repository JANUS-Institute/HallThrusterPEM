"""Module for plume models"""

import numpy as np
from scipy.special import erfi
import logging
import sys
import math
import scipy.integrate

sys.path.append('..')
Q_E = 1.602176634e-19   # Fundamental charge (C)
kB = 1.380649e-23       # Boltzmann constant (J/K)
TORR_2_PA = 133.322
logger = logging.getLogger(__name__)

from utils import ModelRunException


def current_density_model(plume_input, compute_div=False):
    # Load plume inputs
    theta = [plume_input['c0'], plume_input['c1'], plume_input['c2'], plume_input['c3'], plume_input['c4'],
             plume_input['c5']]
    P_B = plume_input['background_pressure_Torr'] / plume_input['p_ref']
    I_B0 = plume_input['I_B0']
    sigma_cex = plume_input['sigma_cex']

    # Make sure prediction locations are same dimension
    r = np.atleast_1d(plume_input['r_m'])
    alpha = np.atleast_1d(plume_input['alpha_deg'])
    assert len(r.shape) == len(alpha.shape) == 1
    assert r.shape[0] == alpha.shape[0]

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
            # Requires alpha = [..., 90] and r=[#]
            if compute_div:
                start_idx = np.argmin(np.abs(alpha-0))  # Start at thruster centerline
                num_int = j_ion.real * np.cos(alpha) * np.sin(alpha)
                den_int = j_ion.real * np.cos(alpha)

                try:
                    cos_div = scipy.integrate.simps(num_int[start_idx:], alpha[start_idx:]) / \
                              scipy.integrate.simps(den_int[start_idx:], alpha[start_idx:])
                except:
                    # Catch any case where the integration = 0 in the denominator
                    logger.warning('Predicted beam current has 0 integrated ion current, setting div_angle=90 deg')
                    cos_div = 0

        except Exception as e:
            raise ModelRunException(f"Exception in plume model: {e}")

        else:
            ret = {'ion_current_density': list(j_ion.real), 'cathode_current_density': float(j_cathode.real)}

            if compute_div:
                ret['divergence_efficiency'] = cos_div**2
                ret['divergence_angle'] = np.arccos(cos_div)

            return ret


def jion_modified(plume_input, compute_div=False):
    # Load plume inputs
    theta = [plume_input['c0'], plume_input['c1'], plume_input['c2'], plume_input['c3'],
             plume_input['c4'], plume_input['c5']]
    P_B = plume_input['background_pressure_Torr'] * TORR_2_PA
    T_B = plume_input['background_temperature_K']
    I_B0 = plume_input['I_B0']              # Amps
    sigma_cex = plume_input['sigma_cex']    # m^2

    # Make sure prediction locations are same dimension
    r = np.atleast_1d(plume_input['r_m'])
    alpha = np.atleast_1d(plume_input['alpha_deg'])
    assert len(r.shape) == len(alpha.shape) == 1
    assert r.shape[0] == alpha.shape[0]

    # Append cathode prediction location
    r = np.append(r, plume_input['r_cathode'])
    alpha = np.append(alpha, plume_input['alpha_cathode']) * np.pi / 180

    # Compute model prediction
    with np.errstate(invalid='raise', divide='raise'):
        try:
            # Neutral density
            n = theta[4] * P_B + theta[5]  # m^-3

            # Divergence angles
            alpha1 = theta[2] * P_B + theta[3]  # Main beam divergence (rad)
            alpha2 = alpha1 / theta[1]          # Scattered beam divergence (rad)

            # Check for case where divergence is less than 0 and return garbage
            if alpha1 <= 0:
                j_ion = np.zeros(r.shape[0] - 1)
                return {'ion_current_density': list(j_ion), 'cathode_current_density': 0}

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
            # Requires alpha = [... 0, ..., 90] and r=[#]
            if compute_div:
                start_idx = np.argmin(np.abs(alpha-0))  # Start at thruster centerline
                num_int = j_ion.real * np.cos(alpha) * np.sin(alpha)
                den_int = j_ion.real * np.cos(alpha)

                try:
                    cos_div = scipy.integrate.simps(num_int[start_idx:], alpha[start_idx:]) / \
                              scipy.integrate.simps(den_int[start_idx:], alpha[start_idx:])
                except:
                    # Catch any case where the integration = 0 in the denominator
                    logger.warning('Predicted beam current has 0 integrated ion current, setting div_angle=90 deg')
                    cos_div = 0

        except Exception as e:
            raise ModelRunException(f"Exception in plume model: {e}")

        else:
            ret = {'ion_current_density': list(j_ion.real), 'cathode_current_density': float(j_cathode.real)}

            if compute_div:
                ret['divergence_efficiency'] = cos_div**2
                ret['divergence_angle'] = np.arccos(cos_div)

            return ret


if __name__ == '__main__':
    inputs = {
        "c0": 0.747388,
        "c1": 0.348462,
        "c2": -20.66,
        "c3": 0.5917,
        "r_cathode": 0.10158337353817158,
        "alpha_cathode": 73.2190961990383,
        "background_pressure_Torr": 1.57e-05,
        "I_B0": 3.6,
        "sigma_cex": 55e-20,
        "r_m": [1]*31,
        "alpha_deg":
            [
                -10.0,
                -5.0,
                0.0,
                5.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                95.0,
                100.0,
                105.0,
                110.0,
                115.0,
                120.0,
                125.0,
                130.0,
                135.0,
                140.0
            ],
        "background_temperature_K": 300
    }

    res = jion_modified(inputs)
    print('done')
