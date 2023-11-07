import numpy as np
from pathlib import Path


def load_thrust():
    """Load thrust data"""
    base_dir = Path(__file__).parent
    data = np.loadtxt(base_dir / 'thrust.csv', delimiter=',', skiprows=1)
    pb = np.log10(data[:, 2, np.newaxis])
    Va = data[:, 0, np.newaxis]
    mdot_a = data[:, 1, np.newaxis]           # Anode flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)
    y = data[:, 3, np.newaxis]/1000           # Thrust (N)
    var_y = (data[:, 4, np.newaxis]*y / 2)**2

    return dict(x=x, y=y, var_y=var_y)
