import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy

sys.path.append('..')

from utils import parse_input_file
from models.cc import cathode_coupling_model_feedforward as cc_model


def test_cc():
    """Test function for cc model"""
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')

    def func(pb, cprime, ui, jT):
        cc_nominal['c_prime'] = cprime
        cc_nominal['avg_ion_velocity'] = ui
        cc_nominal['cathode_current_density'] = jT
        v_cc = np.zeros(len(pb))
        for i, p in enumerate(pb):
            cc_nominal['background_pressure_Torr'] = p
            v_cc[i] = cc_model(cc_nominal)

        return v_cc

    def func2(pb, cprime):
        cc_nominal['c_prime'] = cprime
        v_cc = np.zeros(len(pb))
        for i, p in enumerate(pb):
            cc_nominal['background_pressure_Torr'] = p
            v_cc[i] = cc_model(cc_nominal)

        return v_cc

    p_ref = 1e-05  # Torr
    xdata = np.array([1.67e-6, 4.11e-6, 6.97e-6, 12.3e-6, 15.8e-6, 25.1e-6, 38.2e-6, 55.1e-6])
    ydata = np.array([31.2, 31.95, 32.02, 32.84, 32.83, 33.09, 32.58, 32.1])

    popt, pcov = scipy.optimize.curve_fit(func2, xdata, ydata, p0=0.5, bounds=(0, 1))
    # popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata, p0=[0.5, 24000, 7], bounds=([0, 20000, 1], [1, 30000, 100]))

    N = 100
    PB = np.linspace(0, 6, N) * p_ref
    v_cc = np.zeros(N)
    cc_nominal['c_prime'] = popt
    # cc_nominal['avg_ion_velocity'] = popt[1]
    # cc_nominal['cathode_current_density'] = popt[2]
    for i in range(N):
        cc_nominal['background_pressure_Torr'] = PB[i]
        v_cc[i] = cc_model(cc_nominal)

    plt.figure()
    plt.plot(PB/p_ref, v_cc, '-k')
    plt.errorbar(xdata/p_ref, ydata, yerr=0.3, fmt='ok', capsize=4, markerfacecolor='none')
    plt.xlabel(r'$\bar{P}_B$')
    plt.ylabel(r'$V_{cc}$')
    plt.show()