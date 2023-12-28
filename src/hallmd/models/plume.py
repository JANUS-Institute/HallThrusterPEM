""" `plume.py`

Module for Hall thruster plume models.

Includes
--------
- `plume_feedforward()` - Semi-empirical feedforward plume model.
- `jion_reconstruct()` - Convenience function for reconstructing ion current density profiles from compressed data.
"""
import pickle
from pathlib import Path
from importlib import resources

from amisc.utils import get_logger
import numpy as np
from scipy.special import erfi
from scipy.integrate import simps
from scipy.interpolate import interp1d

from hallmd.utils import ModelRunException
from hallmd.models import config as model_config_dir

Q_E = 1.602176634e-19   # Fundamental charge (C)
kB = 1.380649e-23       # Boltzmann constant (J/K)
TORR_2_PA = 133.322
LOGGER = get_logger(__name__)
CONFIG_DIR = resources.files(model_config_dir)


def plume_feedforward(x: np.ndarray, compress: bool = False,
                      svd_file: str | Path = CONFIG_DIR / 'plume_svd.pkl'):
    """Compute the semi-empirical ion current density ($j_{ion}$) plume model.

    :param x: `(..., xdim)` Plume inputs
    :param compress: Whether to return the dimension-reduced $j_{ion}$ profile
    :param svd_file: Path to `.pkl` SVD data file for compressing the ion current density profile
    :raises ModelRunException: if anything fails
    :returns: `y` - `(..., ydim)` Plume outputs
    """
    # Load plume inputs
    P_B = 10 ** (x[..., 0, np.newaxis]) * TORR_2_PA     # Background pressure (Torr log10)
    c0 = x[..., 1, np.newaxis]                          # Fit coefficients
    c1 = x[..., 2, np.newaxis]
    c2 = x[..., 3, np.newaxis]
    c3 = x[..., 4, np.newaxis]
    c4 = 10 ** (x[..., 5, np.newaxis])
    c5 = 10 ** (x[..., 6, np.newaxis])
    sigma_cex = x[..., 7, np.newaxis] * 1e-20           # Charge-exchange cross-section (m^2)
    r_m = x[..., 8, np.newaxis]                         # Axial distance from thruster exit plane (m)
    I_B0 = x[..., 9, np.newaxis]                        # Total initial ion beam current (A)

    # Load svd params for dimension reduction
    if compress:
        with open(svd_file, 'rb') as fd:
            svd_data = pickle.load(fd)
            vtr = svd_data['vtr']       # (r x M)
            A = svd_data['A']
            A_mu = np.mean(A, axis=0)
            A_std = np.std(A, axis=0)
            r, M = vtr.shape
            ydim = r + 1
    else:
        M = 100
        ydim = M + 1

    # Compute model prediction
    alpha_rad = np.reshape(np.linspace(0, np.pi/2, M), (1,)*len(x.shape[:-1]) + (M,))
    y = np.zeros(x.shape[:-1] + (ydim,))
    try:
        # Neutral density
        n = c4 * P_B + c5  # m^-3

        # Divergence angles
        alpha1 = c2 * P_B + c3  # Main beam divergence (rad)
        alpha2 = alpha1 / c1    # Scattered beam divergence (rad)

        with np.errstate(invalid='ignore', divide='ignore'):
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

        # Set j~0 where alpha1 < 0 (invalid cases)
        j[np.where(alpha1 <= 0)] = 1e-20
        j[np.where(j <= 0)] = 1e-20

        if np.any(abs(j.imag) > 0):
            LOGGER.warning('Predicted beam current has imaginary component.')

        # Calculate divergence angle from https://aip.scitation.org/doi/10.1063/5.0066849
        # Requires alpha = [0, ..., 90] deg
        num_int = j.real * np.cos(alpha_rad) * np.sin(alpha_rad)
        den_int = j.real * np.cos(alpha_rad)

        with np.errstate(divide='ignore'):
            cos_div = simps(num_int, alpha_rad, axis=-1) / simps(den_int, alpha_rad, axis=-1)
            cos_div[cos_div == np.inf] = np.nan

        y[..., 0] = np.arccos(cos_div)  # Divergence angle (rad)

        # Ion current density (A/m^2), in compressed dimension (r) or default dimension (M)
        y[..., 1:] = np.squeeze(vtr @ ((j.real - A_mu) / A_std)[..., np.newaxis], axis=-1) if compress else j.real
        return {'y': y, 'cost': 1}

    except Exception as e:
        raise ModelRunException(f"Exception in plume model: {e}")


def jion_reconstruct(xr: np.ndarray, alpha: np.ndarray = None,
                     svd_file: str | Path = CONFIG_DIR / 'plume_svd.pkl') -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct an ion current density profile, interpolate to `alpha` if provided.

    !!! Warning
        The `svd_file` must be the same as was used when originally compressing the data in `plume_feedforward()`.

    :param xr: `(... r)` The reduced dimension output of `plume_feedforward()`, (just the ion current density)
    :param alpha: `(Nx,)` The alpha grid points to interpolate to (in radians, between -pi/2 and pi/2)
    :param svd_file: Path to `.pkl` SVD data file for compressing the ion current density profile
    :returns: `alpha`, `jion_interp` - `(..., Nx or M)` The reconstructed (and optionally interpolated) jion profile(s),
                    corresponds to `alpha=(0, 90)` deg with `M=100` points by default
    """
    with open(svd_file, 'rb') as fd:
        svd_data = pickle.load(fd)
        vtr = svd_data['vtr']       # (r x M)
        A = svd_data['A']
        A_mu = np.mean(A, axis=0)
        A_std = np.std(A, axis=0)
        r, M = vtr.shape
    alpha_g = np.linspace(0, np.pi/2, M)
    jion_g = np.squeeze(vtr.T @ xr[..., np.newaxis], axis=-1) * A_std + A_mu  # (..., M)

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
