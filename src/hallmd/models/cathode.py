"""Module for cathode models.

Includes:

- `cathode_coupling()` - cathode coupling model with pressure dependence (Jorns 2021)
"""
import numpy as np
from amisc.typing import Dataset

from hallmd.utils import TORR_2_PA

__all__ = ['cathode_coupling']


def cathode_coupling(inputs: Dataset) -> Dataset:
    """Computes cathode coupling voltage dependence on background pressure.

    :param inputs: input arrays - `P_b`, `V_a`, `T_e`, `V_vac`, `Pstar`, `P_T` for background pressure (Torr), discharge
                   voltage (V), electron temperature (eV), vacuum coupling voltage (V), and model parameters P* (Torr)
                   and P_T (Torr).
    :returns outputs: output arrays - `V_cc` for cathode coupling voltage (V).
    """
    # Load inputs
    PB = inputs['P_b'] * TORR_2_PA          # Background Pressure (Torr)
    Va = inputs['V_a']                      # Anode voltage (V)
    Te = inputs['T_e']                      # Electron temperature at the cathode (eV)
    V_vac = inputs['V_vac']                 # Vacuum coupling voltage model parameter (V)
    Pstar = inputs['Pstar'] * TORR_2_PA     # Model parameter P* (Torr)
    PT = inputs['P_T'] * TORR_2_PA          # Model parameter P_T (Torr)

    # Compute cathode coupling voltage
    V_cc = np.atleast_1d(V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB)
    V_cc[V_cc < 0] = 0
    ind = np.where(V_cc > Va)
    V_cc[ind] = np.atleast_1d(Va)[ind]
    return {'V_cc': V_cc}
