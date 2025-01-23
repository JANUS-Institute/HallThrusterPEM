"""Module for cathode models.

Includes:

- `cathode_coupling()` - cathode coupling model with pressure dependence (Jorns 2021)
"""

import json
import random
import string
from pathlib import Path

import numpy as np
from amisc.typing import Dataset

__all__ = ['cathode_coupling']


def cathode_coupling(inputs: Dataset, output_path: str | Path | None = None) -> Dataset:
    """Computes cathode coupling voltage dependence on background pressure.

    :param inputs: input arrays - `P_b`, `V_a`, `T_e`, `V_vac`, `Pstar`, `P_T` for background pressure (Torr), discharge
                   voltage (V), electron temperature (eV), vacuum coupling voltage (V), and model parameters P* (Torr)
                   and P_T (Torr).
    :returns outputs: output arrays - `V_cc` for cathode coupling voltage (V).
    """
    # Load inputs
    PB = np.atleast_1d(inputs['P_b'])  # Background Pressure (Torr)
    Va = np.atleast_1d(inputs['V_a'])  # Anode voltage (V)
    Te = np.atleast_1d(inputs['T_e'])  # Electron temperature at the cathode (eV)
    V_vac = np.atleast_1d(inputs['V_vac'])  # Vacuum coupling voltage model parameter (V)
    Pstar = np.atleast_1d(inputs['Pstar'])  # Model parameter P* (Torr)
    PT = np.atleast_1d(inputs['P_T'])  # Model parameter P_T (Torr)

    # Compute cathode coupling voltage
    V_cc = V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB
    V_cc[V_cc < 0] = 0
    ind = np.where(V_cc > Va)
    V_cc[ind] = np.atleast_1d(Va)[ind]

    # output to file
    # operating conditions (Va, PB) may have several distinct values,
    # while model parameters will be lists containing a single value
    inputs_json = {
        "discharge_voltage_v": Va.tolist(),
        "background_pressure_torr": PB.tolist(),
        "vacuum_coupling_voltage_v": V_vac[0],
        "cathode_electron_temp_ev": Te[0],
        "pstar_torr": Pstar[0],
        "pt_torr": PT[0],
    }
    outputs_json = {"cathode_coupling_voltage_v": V_cc.tolist()}

    fname = "cathode_" + "".join(random.choices(string.digits + string.ascii_letters, k=6)) + ".json"
    if output_path is None:
        output_file = fname
    else:
        output_file = str((Path(output_path) / fname).resolve())

    with open(output_file, "w") as f:
        json.dump({"input": inputs_json, "output": outputs_json}, f)

    return {'V_cc': V_cc}
