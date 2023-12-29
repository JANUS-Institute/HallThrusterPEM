"""Helper functions for loading Diamant 2014 datasets for the SPT-100."""
from importlib import resources

import numpy as np

from hallmd import ExpData
from hallmd.data.spt100 import diamant2014 as base_package

BASE_DIR = resources.files(base_package)


def load_thrust() -> ExpData:
    """Load aerospace and L-3 thrust data."""
    x = np.zeros((0, 3))
    y = np.zeros((0, 1))
    var_y = np.zeros((0, 1))
    for fname in ['thrust_aerospace.csv', 'thrust_L-3.csv']:
        thrust_data = np.loadtxt(BASE_DIR / fname, delimiter=',', skiprows=1)
        pb = np.log10(thrust_data[:, 3, np.newaxis])      # Background pressure near thruster exit (log10 Torr)
        Va = thrust_data[:, 0, np.newaxis]      # Anode voltage (V)
        mdot = thrust_data[:, 1, np.newaxis]    # Total flow rate (mg/s)
        ma2mc = thrust_data[:, 2, np.newaxis]   # Anode to cathode flow ratio
        mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
        x_new = np.concatenate((pb, Va, mdot_a), axis=1)
        thrust = thrust_data[:, 4, np.newaxis]*1e-3  # Thrust (N)
        thrust_var = (thrust_data[:, 5, np.newaxis]*thrust / 2) ** 2    # N^2

        x = np.concatenate((x, x_new), axis=0)  # (PB, Va, mdot_a)
        y = np.concatenate((y, thrust), axis=0)
        var_y = np.concatenate((var_y, thrust_var), axis=0)

    return dict(x=x, y=y, var_y=var_y)


def load_vcc() -> ExpData:
    """Load cathode coupling voltage data from Jorns 2021."""
    data = np.loadtxt(BASE_DIR / 'vcc_jorns2021.csv', delimiter=',', skiprows=1)
    pb = np.log10(data[:, 3, np.newaxis])
    Va = data[:, 0, np.newaxis]
    mdot = data[:, 1, np.newaxis]           # Total flow rate (mg/s)
    ma2mc = data[:, 2, np.newaxis]          # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)
    y = data[:, 4, np.newaxis]              # Coupling voltage (V)
    var_y = (data[:, 5, np.newaxis] / 2)**2

    return dict(x=x, y=y, var_y=var_y)


def load_jion() -> ExpData:
    """Load ion current density profile from L-3."""
    data = np.loadtxt(BASE_DIR / 'jion_L-3.csv', delimiter=',', skiprows=1)
    N = 8   # 8 unique operating conditions for L-3 dataset
    data = np.reshape(data, (N, -1, 8))

    # Load operating conditions
    pb = np.log10(data[:, 0, 3, np.newaxis])
    Va = data[:, 0, 0, np.newaxis]
    mdot = data[:, 0, 1, np.newaxis]        # Total flow rate (mg/s)
    ma2mc = data[:, 0, 2, np.newaxis]       # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)

    # Load coordinates of data
    loc = np.zeros((data.shape[:2] + (2,)))
    loc[..., 0] = data[..., 4]              # Axial location (m)
    loc[..., 1] = data[..., 5]*np.pi/180    # Radial location (rad)

    y = data[..., 6, np.newaxis]*10         # Ion current density (A/m^2)
    var_y = (data[..., 7, np.newaxis] * y / 2) ** 2

    return dict(x=x, loc=loc, y=y, var_y=var_y)
