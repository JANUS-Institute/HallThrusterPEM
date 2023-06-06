"""Module for plume models"""

import numpy as np
from scipy.special import erfi
import logging
import sys
import math
from scipy.integrate import simps
from scipy.interpolate import interp1d
import pickle
from pathlib import Path

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
                    cos_div = simps(num_int[start_idx:], alpha[start_idx:]) / \
                              simps(den_int[start_idx:], alpha[start_idx:])
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
                    cos_div = simps(num_int[start_idx:], alpha[start_idx:]) / \
                              simps(den_int[start_idx:], alpha[start_idx:])
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


def plume_pem(x, alpha=(), compress=True):
    """Compute the ion current density plume model in PEM format
    :param x: (..., xdim) Plume inputs
    :param alpha: Model fidelity indices (none for plume model, since it is analytical)
    :param compress: Whether to return the dimension-reduced jion profile
    :returns y: (..., ydim) Plume outputs
    """
    # Load plume inputs
    P_B = x[..., 0, np.newaxis] * TORR_2_PA
    c0 = x[..., 1, np.newaxis]
    c1 = x[..., 2, np.newaxis]
    c2 = x[..., 3, np.newaxis]
    c3 = x[..., 4, np.newaxis]
    c4 = x[..., 5, np.newaxis]
    c5 = x[..., 6, np.newaxis]
    sigma_cex = x[..., 7, np.newaxis]   # m^2
    r_m = x[..., 8, np.newaxis]         # m
    I_B0 = x[..., 9, np.newaxis]        # A

    # Load svd params for dimension reduction
    if compress:
        with open(Path(__file__).parent / 'plume_svd.pkl', 'rb') as fd:
            svd_data = pickle.load(fd)
            vtr = svd_data['vtr']       # (r x M)
            r, M = vtr.shape
            ydim = r+2
    else:
        ydim = 102
        M = 100

    # Compute model prediction
    alpha_rad = np.reshape(np.linspace(0, np.pi/2, M), (1,)*len(x.shape[:-1]) + (M,))
    y = np.zeros(x.shape[:-1] + (ydim,))
    try:
        # Neutral density
        n = c4 * P_B + c5  # m^-3

        # Divergence angles
        alpha1 = c2 * P_B + c3  # Main beam divergence (rad)
        alpha2 = alpha1 / c1    # Scattered beam divergence (rad)

        with np.errstate(invalid='ignore'):
            A1 = (1 - c0) / ((np.pi ** (3 / 2)) / 2 * alpha1 * np.exp(-(alpha1 / 2)**2) *
                             (2 * erfi(alpha1 / 2) + erfi((np.pi * 1j - (alpha1 ** 2)) / (2 * alpha1)) -
                              erfi((np.pi * 1j + (alpha1 ** 2)) / (2 * alpha1))))
            A2 = c0 / ((np.pi ** (3 / 2)) / 2 * alpha2 * np.exp(-(alpha2 / 2)**2) *
                       (2 * erfi(alpha2 / 2) + erfi((np.pi * 1j - (alpha2 ** 2)) / (2 * alpha2)) -
                        erfi((np.pi * 1j + (alpha2 ** 2)) / (2 * alpha2))))

        I_B = I_B0 * np.exp(-r_m*n*sigma_cex)
        j_beam = (I_B / r_m ** 2) * A1 * np.exp(-(alpha_rad / alpha1) ** 2)
        j_scat = (I_B / r_m ** 2) * A2 * np.exp(-(alpha_rad / alpha2) ** 2)
        j_cex = I_B0 * (1 - np.exp(-r_m*n*sigma_cex)) / (2 * np.pi * r_m ** 2)
        j = j_beam + j_scat + j_cex

        # Check for case where divergence is less than 0 and return nan
        neg_idx = np.where(alpha1[..., 0] <= 0)
        j[neg_idx, :] = np.nan

        if np.any(abs(j.imag) > 0):
            logger.warning('Predicted beam current has imaginary component')

        # Calculate divergence angle from https://aip.scitation.org/doi/10.1063/5.0066849
        # Requires alpha = [0, ..., 90]
        num_int = j.real * np.cos(alpha_rad) * np.sin(alpha_rad)
        den_int = j.real * np.cos(alpha_rad)

        with np.errstate(divide='ignore'):
            cos_div = simps(num_int, alpha_rad, axis=-1) / simps(den_int, alpha_rad, axis=-1)
            cos_div[cos_div == np.inf] = np.nan

        y[..., 0] = cos_div**2          # Divergence efficiency
        y[..., 1] = np.arccos(cos_div)  # Divergence angle (rad)

        # Ion current density (A/m^2), in compressed dimension (r) or default dimension (M)
        y[..., 2:] = np.squeeze(vtr @ np.log10(j[..., np.newaxis].real), axis=-1) if compress else j.real
        return y

    except Exception as e:
        raise ModelRunException(f"Exception in plume model: {e}")


def jion_reconstruct(xr, alpha=None):
    """Reconstruct an ion current density profile, interpolate to alpha if provided
    :param xr: (... r) The reduced dimension output of plume_pem (just the ion current density)
    :param alpha: (Nx,) The alpha grid points to interpolate to (in radians, between -pi/2 and pi/2)
    :returns alpha, jion_interp: (..., Nx or M) The reconstructed (and potentially interpolated) jion profile(s),
                    corresponds to alpha=(0, 90) deg with M=100 points by default
    """
    with open(Path(__file__).parent / 'plume_svd.pkl', 'rb') as fd:
        svd_data = pickle.load(fd)
        vtr = svd_data['vtr']       # (r x M)
        r, M = vtr.shape
    alpha_g = np.linspace(0, np.pi/2, M)
    jion_g = 10 ** np.squeeze(vtr.T @ xr[..., np.newaxis], axis=-1)  # (..., M)

    # Do interpolation
    if alpha is not None:
        # Extend to range (-90, 90) deg
        alpha_g2 = np.concatenate((-np.flip(alpha_g)[:-1], alpha_g))                     # (2M-1,)
        jion_g2 = np.concatenate((np.flip(jion_g, axis=-1)[..., :-1], jion_g), axis=-1)  # (..., 2M-1)

        f = interp1d(alpha_g2, jion_g2, axis=-1)
        jion_interp = f(alpha)  # (..., Nx)
        return alpha, jion_interp
    else:
        return alpha_g, jion_g
