"""Helper functions for loading UM datasets for the H9."""
from importlib import resources

import numpy as np

from hallmd import ExpData
# from hallmd.data.spt100 import diamant2014 as base_package

# BASE_DIR = resources.files(base_package)
BASE_DIR = '/home/morag/h9-data'

def load_thrust() -> ExpData: # no UM thrust data
    """Load aerospace and L-3 thrust data."""
    x = np.zeros((0, 3))
    y = np.zeros((0,))
    var_y = np.zeros((0,))
    thrust_data = np.loadtxt(BASE_DIR + '/GT/thrust_GT.csv', delimiter=',', skiprows=1)
    pb = np.log10(thrust_data[:, 3, np.newaxis])        # Background pressure near thruster exit (log10 Torr)
    Va = thrust_data[:, 0, np.newaxis]                  # Anode voltage (V)
    mdot = thrust_data[:, 1, np.newaxis]                # Total flow rate (mg/s)
    ma2mc = thrust_data[:, 2, np.newaxis]               # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot                 # Anode mass flow rate (mg/s)
    x_new = np.concatenate((pb, Va, mdot_a), axis=1)
    thrust = thrust_data[:, 4]*1e-3                     # Thrust (N)
    thrust_var = (thrust_data[:, 5]*thrust / 2) ** 2    # N^2

    x = np.concatenate((x, x_new), axis=0)  # (PB, Va, mdot_a)
    y = np.concatenate((y, thrust), axis=0)
    var_y = np.concatenate((var_y, thrust_var), axis=0)

    return dict(x=x, y=y, var_y=var_y)


def load_vcc() -> ExpData:
    """Load cathode coupling voltage data from Jorns 2021."""
    data = np.loadtxt(BASE_DIR + '/UM/vcc_UMH9.csv', delimiter=',', skiprows=1)
    pb = np.log10(data[:, 3, np.newaxis])
    Va = data[:, 0, np.newaxis]
    mdot = data[:, 1, np.newaxis]           # Total flow rate (mg/s)
    ma2mc = data[:, 2, np.newaxis]          # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)
    y = data[:, 4]                          # Coupling voltage (V)
    var_y = (data[:, 5] / 2)**2

    return dict(x=x, y=y, var_y=var_y)


def load_jion() -> ExpData:
    """Load ion current density profile from L-3."""
    data = np.loadtxt(BASE_DIR + '/UM/jion_UMH9.csv', delimiter=',', skiprows=1)
    N = 12   # 8 unique operating conditions for L-3 dataset
    data = np.reshape(data, (N, -1, 8))

    # Load operating conditions
    pb = np.log10(data[:, 0, 3, np.newaxis])
    Va = data[:, 0, 0, np.newaxis]
    mdot = data[:, 0, 1, np.newaxis]        # Total flow rate (mg/s)
    ma2mc = data[:, 0, 2, np.newaxis]       # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)

    # Load coordinates of data (assume they are the same for all operating conditions)
    loc = np.zeros((data.shape[1], 2))
    loc[:, 0] = data[0, :, 4]                # Axial location (m)
    loc[:, 1] = (data[0, :, 5]-90)*np.pi/180 # Radial location (rad)

    y = data[..., 6]*10                     # Ion current density (A/m^2)
    var_y = (data[..., 7] * y / 2) ** 2

    # Only keep less than 90 deg data
    keep_idx = loc[:, 1] < np.pi / 2
    y = y[:, keep_idx].reshape((N, -1))
    var_y = var_y[:, keep_idx].reshape((N, -1))
    loc = loc[keep_idx, :].reshape((-1, 2))
    return dict(x=x, loc=loc, y=y, var_y=var_y)


def load_uion() -> ExpData:
    """Load peak ion velocity profiles along channel centerline."""
    data = np.loadtxt(BASE_DIR + '/UM/uion_UMH9.csv', delimiter=',', skiprows=1)
    N = 5   # unique operating conditions
    data = np.reshape(data, (N, -1, 7))

    # Load operating conditions
    pb = np.log10(data[:, 0, 3, np.newaxis])
    Va = data[:, 0, 0, np.newaxis]
    mdot = data[:, 0, 1, np.newaxis]        # Total flow rate (mg/s)
    ma2mc = data[:, 0, 2, np.newaxis]       # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)

    # Load coordinates of data
    loc = np.zeros((data.shape[1], ))
    loc[:] = data[0, :, 4] * 1e-3             # Axial location from anode along channel centerline (m)

    y = data[..., 5]                    # Time-avg peak Xe+ velocity (m/s)
    #var_y = ((y * data[..., 6]) / 2) ** 2
    var_y = (500 * np.ones(data[..., 6].shape) / 2) ** 2

    return dict(x=x, loc=loc, y=y, var_y=var_y)


def load_gt_thrust() -> ExpData: # no UM thrust data
    """Load aerospace and L-3 thrust data."""
    data = np.loadtxt(BASE_DIR + '/GT/thrust_GT.csv', delimiter=',', skiprows=1)
    pb = np.log10(data[:, 4, np.newaxis])        # Background pressure near thruster exit (log10 Torr)
    Va = data[:, 0, np.newaxis]                  # Anode voltage (V)
    mdot = data[:, 1, np.newaxis]                # Total flow rate (mg/s)
    ma2mc = data[:, 2, np.newaxis]               # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot                 # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)
    y = data[:, 5]                          # Thrust (N)
    var_y = (0.009*y / 2) ** 2    # N^2

    return dict(x=x, y=y, var_y=var_y)


def load_gt_jion() -> ExpData:
    """Load ion current density profile from L-3."""
    data = np.loadtxt(BASE_DIR + '/GT/jion_GT.csv', delimiter=',', skiprows=2)
    N = 3   # 11 operating conditions for L-3 dataset
    data = np.reshape(data, (N, -1, 11))

    # Load operating conditions
    pb = np.log10(data[:, 0, 4, np.newaxis])
    Va = data[:, 0, 0, np.newaxis]
    mdot = data[:, 0, 1, np.newaxis]        # Total flow rate (mg/s)
    ma2mc = data[:, 0, 2, np.newaxis]       # Anode to cathode flow ratio
    mdot_a = ma2mc / (1 + ma2mc) * mdot     # Anode mass flow rate (mg/s)
    x = np.concatenate((pb, Va, mdot_a), axis=1)

    # Load coordinates of data (assume they are the same for all operating conditions)
    loc = np.zeros((data.shape[1], 2))
    loc[:, 0] = data[0, :, 5]               # Axial location (m)
    loc[:, 1] = data[0, :, 6]*np.pi/180     # Radial location (rad)

    y = data[..., 8]                     # Ion current density (A/m^2)
    var_y = (0.2 * y / 2) ** 2

    # Only keep less than 90 deg data
    keep_idx = loc[:, 1] < np.pi / 2
    y = y[:, keep_idx].reshape((N, -1))
    var_y = var_y[:, keep_idx].reshape((N, -1))
    loc = loc[keep_idx, :].reshape((-1, 2))

    return dict(x=x, loc=loc, y=y, var_y=var_y)
