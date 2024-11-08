"""Module for Hall thruster plume models.

Includes:

- `current_density()` - Semi-empirical ion current density model with $1/r^2$ Gaussian beam.
"""
import numpy as np
from scipy.special import erfi
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from amisc.typing import Dataset
from amisc.utils import get_logger

from hallmd.utils import TORR_2_PA


LOGGER = get_logger(__name__)


def current_density(inputs: Dataset, j_ion_coords: np.ndarray = None):
    """Compute the semi-empirical ion current density ($j_{ion}$) plume model over a 90 deg sweep, with 0 deg at
    thruster centerline. Also compute the plume divergence angle.

    :param inputs: input arrays - `P_b`, `c0`, `c1`, `c2`, `c3`, `c4`, `c5`, `sigma_cex`, `r_m`, `I_B0` for background
                   pressure (Torr), plume fit coefficients, charge-exchange cross-section ($m^2$), radial distance
                   from thruster exit plane (m), and total initial ion beam current (A).
    :param j_ion_coords: `(M,)` The angles at which to compute the ion current density, in radians. Defaults to
                         90 deg sweep from 0 deg to 90 deg about thruster centerline in 1 deg increments (91 points).
                         If provided, must be between -90 and 90 deg.
    :returns outputs: output arrays - `j_ion` for ion current density ($A/m^2$) at the `j_ion_coords` locations,
                                       and `div_angle` in radians for the divergence angle of the plume.
    """
    # Load plume inputs
    P_B = inputs['P_b'] * TORR_2_PA     # Background pressure (Torr)
    c0 = inputs['c0']                   # Fit coefficients
    c1 = inputs['c1']
    c2 = inputs['c2']
    c3 = inputs['c3']
    c4 = inputs['c4']
    c5 = inputs['c5']
    sigma_cex = inputs['sigma_cex']     # Charge-exchange cross-section (m^2)
    r_m = inputs['r_m']                 # Axial distance from thruster exit plane (m)
    I_B0 = inputs['I_B0']               # Total initial ion beam current (A)

    # 90 deg angle sweep for ion current density
    alpha_rad = np.linspace(0, np.pi/2, 91)

    # Neutral density
    n = c4 * P_B + c5  # m^-3

    # Divergence angles
    alpha1 = np.atleast_1d(c2 * P_B + c3)  # Main beam divergence (rad)
    alpha1[alpha1 > np.pi/2] = np.pi/2
    alpha2 = alpha1 / c1                   # Scattered beam divergence (rad)

    with np.errstate(invalid='ignore', divide='ignore'):
        A1 = (1 - c0) / ((np.pi ** (3 / 2)) / 2 * alpha1 * np.exp(-(alpha1 / 2)**2) *
                         (2 * erfi(alpha1 / 2) + erfi((np.pi * 1j - (alpha1 ** 2)) / (2 * alpha1)) -
                          erfi((np.pi * 1j + (alpha1 ** 2)) / (2 * alpha1))))
        A2 = c0 / ((np.pi ** (3 / 2)) / 2 * alpha2 * np.exp(-(alpha2 / 2)**2) *
                   (2 * erfi(alpha2 / 2) + erfi((np.pi * 1j - (alpha2 ** 2)) / (2 * alpha2)) -
                    erfi((np.pi * 1j + (alpha2 ** 2)) / (2 * alpha2))))
        I_B = I_B0 * np.exp(-r_m * n * sigma_cex)

        base_density = np.atleast_1d(I_B / r_m ** 2)[..., np.newaxis]
        j_beam = base_density * A1[..., np.newaxis] * np.exp(-(alpha_rad / alpha1[..., np.newaxis]) ** 2)
        j_scat = base_density * A2[..., np.newaxis] * np.exp(-(alpha_rad / alpha2[..., np.newaxis]) ** 2)
        j_cex = I_B0 * (1 - np.exp(-r_m * n * sigma_cex)) / (2 * np.pi * r_m ** 2)
        j_ion = j_beam + j_scat + np.atleast_1d(j_cex)[..., np.newaxis]  # (..., 91) the current density 1d profile

    # Set j~0 where alpha1 < 0 (invalid cases)
    j_ion[np.where(alpha1 <= 0)] = 1e-20
    j_ion[np.where(j_ion <= 0)] = 1e-20

    if np.any(abs(j_ion.imag) > 0):
        LOGGER.warning('Predicted beam current has non-zero imaginary component.')
    j_ion = j_ion.real

    # Calculate divergence angle from https://aip.scitation.org/doi/10.1063/5.0066849
    # Requires alpha = [0, ..., 90] deg, from thruster exit-plane to thruster centerline (need to flip)
    num_int = np.flip(j_ion - j_cex, axis=-1) * np.cos(alpha_rad) * np.sin(alpha_rad)
    den_int = np.flip(j_ion - j_cex, axis=-1) * np.cos(alpha_rad)

    with np.errstate(divide='ignore'):
        cos_div = simpson(num_int, x=alpha_rad, axis=-1) / simpson(den_int, x=alpha_rad, axis=-1)
        cos_div = np.atleast_1d(cos_div)
        cos_div[cos_div == np.inf] = np.nan

    div_angle = np.arccos(cos_div)  # Divergence angle (rad)

    # Interpolate to requested angles
    if j_ion_coords is not None:
        # Extend to range (-90, 90) deg
        alpha_grid = np.concatenate((-np.flip(alpha_rad)[:-1], alpha_rad))               # (2M-1,)
        jion_grid = np.concatenate((np.flip(j_ion, axis=-1)[..., :-1], j_ion), axis=-1)  # (..., 2M-1)

        f = interp1d(alpha_grid, jion_grid, axis=-1)
        j_ion = f(j_ion_coords)  # (..., num_pts)

    return {'j_ion': j_ion, 'div_angle': div_angle}
