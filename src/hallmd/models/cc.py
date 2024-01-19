""" `cc.py`

Module for cathode coupling models.

Includes
--------
- `cc_feedforward()` - only feedforward cathode coupling model with pressure dependence (Jorns 2021)
- `cc_feedback()` - same model but modified to include feedback from the plume
"""
import numpy as np

Q_E = 1.602176634e-19   # Fundamental charge (C)
kB = 1.380649e-23       # Boltzmann constant (J/K)
TORR_2_PA = 133.322


def cc_feedforward(x: np.ndarray):
    """Compute cathode coupling model with no feedback interactions.

    :param x: `(..., xdim)`, Cathode model inputs
    :returns y: `(..., ydim)`, Cathode model outputs
    """
    # Load inputs
    x = np.atleast_1d(x)
    PB = 10 ** (x[..., 0, np.newaxis]) * TORR_2_PA      # Background Pressure (log10 Torr)
    Va = x[..., 1, np.newaxis]                          # Anode voltage
    Te = x[..., 2, np.newaxis]                          # Electron temperature (eV)
    V_vac = x[..., 3, np.newaxis]                       # Vacuum coupling voltage (V)
    Pstar = x[..., 4, np.newaxis] * 1e-6 * TORR_2_PA    # Model parameter P*
    PT = x[..., 5, np.newaxis] * 1e-6 * TORR_2_PA       # Model parameter P_T

    # Compute cathode coupling voltage
    y = V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB
    y[y < 0] = 0
    ind = np.where(y > Va)
    y[ind] = Va[ind]
    return {'y': y, 'cost': 1}


def cc_feedback(cc_input):
    """Cathode coupling with feedback interactions with the plume.

    :param cc_input: the named model inputs in a dictionary
    :returns: `dict(cathode_potential=res)`
    """
    # Load cathode inputs
    Te = cc_input['cathode_electron_temp_eV']
    V_vac = cc_input['V_vac']
    Pstar = cc_input['Pstar'] * TORR_2_PA
    c_prime = cc_input['c_prime']
    PB = cc_input['background_pressure_Torr'] * TORR_2_PA
    TB = cc_input['background_temperature_K']
    ui_avg = cc_input['avg_ion_velocity']           # avg ion exit velocity from thruster
    ji_T = cc_input['cathode_current_density']      # total current density at cathode location
    Va = cc_input['anode_potential']

    # Equation 12 in Jorns and Byrne, Plasma Sources Sci. Technol. 30 (2021) 015012
    n_e = ji_T / (Q_E * ui_avg)
    PT = (n_e*kB*TB) / c_prime
    V_cc = V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB

    # Threshold between 0 and anode voltage
    V_cc = min(Va, max(0, V_cc))

    return {'cathode_potential': V_cc}
