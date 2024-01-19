"""Helper functions for loading data from Sankovic 1993 for the SPT-100."""
from importlib import resources

import numpy as np

from hallmd import ExpData
from hallmd.data.spt100 import sankovic1993 as base_package

BASE_DIR = resources.files(base_package)


def load_thrust() -> ExpData:
    """Load thrust data."""
    data = np.loadtxt(BASE_DIR / 'thrust.csv', delimiter=',', skiprows=1)
    pb = np.log10(data[:, 2, np.newaxis])
    Va = data[:, 0, np.newaxis]
    mdot_a = data[:, 1, np.newaxis]             # Anode flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)
    y = data[:, 3]/1000                         # Thrust (N)
    var_y = (data[:, 4]*y / 2)**2

    return dict(x=x, y=y, var_y=var_y)
