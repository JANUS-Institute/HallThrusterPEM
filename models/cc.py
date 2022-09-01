# Module for cathode coupling models

import math
import logging
import sys

logger = logging.getLogger(__name__)
sys.path.append('..')

from utils import data_load, data_write


def cathode_coupling_model(cc_input='cc_input.json', thruster_input=None):
    # Load cathode inputs
    logger.info('Running cathode coupling model')
    inputs = data_load(cc_input)
    PT = inputs['parameters']['PT_norm']
    Pstar = inputs['parameters']['Pstar_norm']
    V_vac = inputs['parameters']['V_vac']
    Te = inputs['parameters']['cathode_electron_temp_eV']
    PB = inputs['design']['PB_norm']

    # Equation 12 in Jorns and Byrne, Plasma Sources Sci. Technol. 30 (2021) 015012
    V_cc = V_vac + Te * math.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB

    output_data = {"V_cc": V_cc}
    data_write(output_data, 'cc_output.json')

    return V_cc
