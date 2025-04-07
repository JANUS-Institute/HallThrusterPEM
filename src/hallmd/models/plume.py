"""Module for Hall thruster plume models.

Includes:

- `current_density()` - Semi-empirical ion current density model with $1/r^2$ Gaussian beam.
"""

import numpy as np
from amisc.typing import Dataset
from amisc.utils import get_logger
from scipy.integrate import simpson
from scipy.special import erfi

from hallmd.utils import TORR_2_PA

__all__ = ['current_density']

LOGGER = get_logger(__name__)


def current_density(inputs: Dataset, sweep_radius: float | list = 1.0):
    """Compute the semi-empirical ion current density ($j_{ion}$) plume model over a 90 deg sweep, with 0 deg at
    thruster centerline. Also compute the plume divergence angle. Will return the ion current density at 91 points,
    from 0 to 90 deg in 1 deg increments. The angular locations are returned as `j_ion_coords` in radians.

    :param inputs: input arrays - `P_b`, `c0`, `c1`, `c2`, `c3`, `c4`, `c5`, `sigma_cex`, `I_B0` for background
                   pressure (Torr), plume fit coefficients, charge-exchange cross-section ($m^2$),
                   and total initial ion beam current (A). If `T` is provided, then
                   also compute corrected thrust using the divergence angle.
    :param sweep_radius: the location(s) at which to compute the ion current density 90 deg sweep, in units of radial
                         distance (m) from the thruster exit plane. If multiple locations are provided, then the
                         returned $j_{ion}$ array's last dimension will match the length of `sweep_radius`. Defaults to
                         1 meter.
    :returns outputs: output arrays - `j_ion` for ion current density ($A/m^2$) at the `j_ion_coords` locations,
                                       and `div_angle` in radians for the divergence angle of the plume. Optionally,
                                       `T_c` for corrected thrust (N) if `T` is provided in the inputs.
    """
    # Load plume inputs
    P_B = inputs['P_b'] * TORR_2_PA  # Background pressure (Torr)
    c0 = inputs['c0']  # Fit coefficients (-)
    c1 = inputs['c1']  # (-)
    c2 = inputs['c2']  # (rad/Pa)
    c3 = inputs['c3']  # (rad)
    c4 = inputs['c4']  # (m^-3/Pa)
    c5 = inputs['c5']  # (m^-3)
    sigma_cex = inputs['sigma_cex']  # Charge-exchange cross-section (m^2)
    I_B0 = inputs['I_B0']  # Total initial ion beam current (A)
    thrust = inputs.get('T', None)  # Thrust (N)
    sweep_radius = np.atleast_1d(sweep_radius)

    # 90 deg angle sweep for ion current density
    alpha_rad = np.linspace(0, np.pi / 2, 91)

    # Neutral density
    n = c4 * P_B + c5  # m^-3

    # Divergence angles
    alpha1 = np.atleast_1d(c2 * P_B + c3)  # Main beam divergence (rad)
    alpha1[alpha1 > np.pi / 2] = np.pi / 2
    alpha2 = alpha1 / c1  # Scattered beam divergence (rad)

    with np.errstate(invalid='ignore', divide='ignore'):
        A1 = (1 - c0) / (
            (np.pi ** (3 / 2))
            / 2
            * alpha1
            * np.exp(-((alpha1 / 2) ** 2))
            * (
                2 * erfi(alpha1 / 2)
                + erfi((np.pi * 1j - (alpha1**2)) / (2 * alpha1))
                - erfi((np.pi * 1j + (alpha1**2)) / (2 * alpha1))
            )
        )
        A2 = c0 / (
            (np.pi ** (3 / 2))
            / 2
            * alpha2
            * np.exp(-((alpha2 / 2) ** 2))
            * (
                2 * erfi(alpha2 / 2)
                + erfi((np.pi * 1j - (alpha2**2)) / (2 * alpha2))
                - erfi((np.pi * 1j + (alpha2**2)) / (2 * alpha2))
            )
        )
        # Broadcast over angles and radii (..., a, r)
        A1 = np.expand_dims(A1, axis=(-1, -2))  # (..., 1, 1)
        A2 = np.expand_dims(A2, axis=(-1, -2))
        alpha1 = np.expand_dims(alpha1, axis=(-1, -2))
        alpha2 = np.expand_dims(alpha2, axis=(-1, -2))
        I_B0 = np.expand_dims(I_B0, axis=(-1, -2))
        n = np.expand_dims(n, axis=(-1, -2))
        sigma_cex = np.expand_dims(sigma_cex, axis=(-1, -2))

        decay = np.exp(-sweep_radius * n * sigma_cex)  # (..., 1, r)
        j_cex = I_B0 * (1 - decay) / (2 * np.pi * sweep_radius**2)

        base_density = I_B0 * decay / sweep_radius**2
        j_beam = base_density * A1 * np.exp(-((alpha_rad[..., np.newaxis] / alpha1) ** 2))
        j_scat = base_density * A2 * np.exp(-((alpha_rad[..., np.newaxis] / alpha2) ** 2))

        j_ion = j_beam + j_scat + j_cex  # (..., 91, r) the current density 1d profile at r radial locations

    # Set j~0 where alpha1 < 0 (invalid cases)
    invalid_idx = np.logical_or(np.any(alpha1 <= 0, axis=(-1, -2)), np.any(j_ion <= 0, axis=(-1, -2)))
    j_ion[invalid_idx, ...] = 1e-20
    j_cex[invalid_idx, ...] = 1e-20

    if np.any(abs(j_ion.imag) > 0):
        LOGGER.warning('Predicted beam current has non-zero imaginary component.')
    j_ion = j_ion.real

    # Calculate divergence angle from https://aip.scitation.org/doi/10.1063/5.0066849
    # Requires alpha = [0, ..., 90] deg, from thruster exit-plane to thruster centerline (need to flip)
    # do j_beam + j_scat instead of j_ion - j_cex to avoid catastrophic loss of precision when
    # j_beam and j_scat << j_cex
    j_non_cex = np.flip((j_beam + j_scat).real, axis=-2)
    den_integrand = j_non_cex * np.cos(alpha_rad[..., np.newaxis])
    num_integrand = den_integrand * np.sin(alpha_rad[..., np.newaxis])

    with np.errstate(divide='ignore', invalid='ignore'):
        num = simpson(num_integrand, x=alpha_rad, axis=-2)
        den = simpson(den_integrand, x=alpha_rad, axis=-2)
        cos_div = np.atleast_1d(num / den)
        cos_div[cos_div == np.inf] = np.nan

    div_angle = np.arccos(cos_div)  # Divergence angle (rad) - (..., r)

    # Squeeze last dim if only a single radius was passed
    if sweep_radius.shape[0] == 1:
        j_ion = np.squeeze(j_ion, axis=-1)
        div_angle = np.squeeze(div_angle, axis=-1)

    ret = {'j_ion': j_ion, 'div_angle': div_angle}

    if thrust is not None:
        thrust_corrected = np.expand_dims(thrust, axis=-1) * cos_div
        if sweep_radius.shape[0] == 1:
            thrust_corrected = np.squeeze(thrust_corrected, axis=-1)
        ret['T_c'] = thrust_corrected

    # Interpolate to requested angles
    # if j_ion_coords is not None:
    #     # Extend to range (-90, 90) deg
    #     alpha_grid = np.concatenate((-np.flip(alpha_rad)[:-1], alpha_rad))               # (2M-1,)
    #     jion_grid = np.concatenate((np.flip(j_ion, axis=-1)[..., :-1], j_ion), axis=-1)  # (..., 2M-1)
    #
    #     f = interp1d(alpha_grid, jion_grid, axis=-1)
    #     j_ion = f(j_ion_coords)  # (..., num_pts)

    # Broadcast coords to same loop shape as j_ion (all use the same coords -- store in object array)
    last_axis = -1 if sweep_radius.shape[0] == 1 else -2
    j_ion_coords = np.empty(j_ion.shape[:last_axis], dtype=object)
    for index in np.ndindex(j_ion.shape[:last_axis]):
        j_ion_coords[index] = alpha_rad

    ret['j_ion_coords'] = j_ion_coords

    return ret
