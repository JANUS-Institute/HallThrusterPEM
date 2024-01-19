"""Helper functions for loading data from MacDonald 2019 for the SPT-100."""
from importlib import resources

import numpy as np

from hallmd import ExpData
from hallmd.data.spt100 import macdonald2019 as base_package

BASE_DIR = resources.files(base_package)


def load_uion() -> ExpData:
    """Load peak ion velocity profiles along channel centerline."""
    data = np.loadtxt(BASE_DIR / 'uion.csv', delimiter=',', skiprows=1)
    N = 3   # unique operating conditions
    data = np.reshape(data, (N, -1, 6))

    # Load operating conditions
    pb = np.log10(data[:, 0, 2, np.newaxis])
    Va = data[:, 0, 0, np.newaxis]
    mdot_a = data[:, 0, 1, np.newaxis]  # Anode flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)

    # Load coordinates of data
    loc = np.zeros((data.shape[1], ))
    loc[:] = data[0, :, 3]              # Axial location from anode along channel centerline (m)

    y = data[..., 4]                    # Time-avg peak Xe+ velocity (m/s)
    var_y = (data[..., 5] / 2) ** 2

    return dict(x=x, loc=loc, y=y, var_y=var_y)
