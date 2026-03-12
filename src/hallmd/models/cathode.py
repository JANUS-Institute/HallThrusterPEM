"""Module for cathode models.

Includes:

- `cathode_coupling()` - cathode coupling model with pressure dependence (Jorns 2021)
"""
from typing import cast

import numpy as np
from pem_core.constants import TORR_2_PA
from pem_core.types import Dataset

__all__ = ['cathode_coupling']


def cathode_coupling(inputs: Dataset) -> Dataset:
    """Computes cathode coupling voltage dependence on background pressure.

    :param inputs: input arrays - `P_b`, `V_a`, `T_e`, `V_vac`, `Pstar`, `P_T` for background pressure (Torr), discharge
                   voltage (V), electron temperature (eV), vacuum coupling voltage (V), and model parameters P* (Torr)
                   and P_T (Torr).
    :returns outputs: output arrays - `V_cc` for cathode coupling voltage (V).
    """
    input_dict = cast(dict, inputs)
    # Load inputs
    PB = input_dict['P_b'] * TORR_2_PA          # Background Pressure (Torr)
    Va = input_dict['V_a']                      # Anode voltage (V)
    Te = input_dict['T_e']                      # Electron temperature at the cathode (eV)
    V_vac = input_dict['V_vac']                 # Vacuum coupling voltage model parameter (V)
    Pstar = input_dict['Pstar'] * TORR_2_PA     # Model parameter P* (Torr)
    PT = input_dict['P_T'] * TORR_2_PA          # Model parameter P_T (Torr)

    # Compute cathode coupling voltage
    V_cc = np.atleast_1d(V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB)
    V_cc[V_cc < 0] = 0
    ind = np.where(V_cc > Va)
    V_cc[ind] = np.atleast_1d(Va)[ind]
    return cast(Dataset, {'V_cc': V_cc})
