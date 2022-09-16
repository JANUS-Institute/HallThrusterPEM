# Module for cathode coupling models

import math
import logging

Q_E = 1.602176634e-19   # Fundamental charge (C)
kB = 1.380649e-23       # Boltzmann constant (J/K)
TORR_2_PA = 133.322
logger = logging.getLogger(__name__)


def cathode_coupling_model(cc_input):
    # Load cathode inputs
    logger.info('Running cathode coupling model')
    Te = cc_input['cathode_electron_temp_eV']
    V_vac = cc_input['V_vac']
    Pstar = cc_input['Pstar'] * TORR_2_PA
    c_prime = cc_input['c_prime']
    PB = cc_input['background_pressure_Torr'] * TORR_2_PA
    TB = cc_input['background_temperature_K']
    ui_avg = cc_input['ion_velocity']           # avg ion exit velocity from thruster
    ji_T = cc_input['ion_current_density']      # total current density at cathode location

    # Equation 12 in Jorns and Byrne, Plasma Sources Sci. Technol. 30 (2021) 015012
    n_e = ji_T / (Q_E * ui_avg)
    PT = (n_e*kB*TB) / c_prime
    V_cc = V_vac + Te * math.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB

    return V_cc
