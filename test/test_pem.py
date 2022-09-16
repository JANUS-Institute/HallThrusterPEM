# Standard imports
import matplotlib.pyplot as plt
import sys
import numpy as np
import logging
import scipy.optimize

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.cc import cathode_coupling_model as cc_model
from models.thruster import hall_thruster_jl_model as thruster_model
from models.plume import current_density_model as plume_model
from utils import parse_input_file


def test_cc():
    """Test function for cc model"""
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')

    def func(pb, cprime, ui, jT):
        cc_nominal['c_prime'] = cprime
        cc_nominal['ion_velocity'] = ui
        cc_nominal['ion_current_density'] = jT
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
    # cc_nominal['ion_velocity'] = popt[1]
    # cc_nominal['ion_current_density'] = popt[2]
    for i in range(N):
        cc_nominal['background_pressure_Torr'] = PB[i]
        v_cc[i] = cc_model(cc_nominal)

    plt.figure()
    plt.plot(PB/p_ref, v_cc, '-k')
    plt.errorbar(xdata/p_ref, ydata, yerr=0.3, fmt='ok', capsize=4, markerfacecolor='none')
    plt.xlabel(r'$\bar{P}_B$')
    plt.ylabel(r'$V_{cc}$')
    plt.show()


def run_nominal():
    # Run cathode model
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
    v_cc = cc_model(cc_nominal)

    # Run thruster model
    thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
    thruster_nominal['cathode_potential'] = v_cc
    ui_avg, I_B0 = thruster_model(thruster_nominal)

    # Run plume model
    plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')
    plume_nominal['I_B0'] = I_B0
    N = 50
    r, alpha, j_ion, j_cathode = plume_model(plume_nominal, N=N)

    # Loop back to cc_model (NOT FOR FEEDFORWARD MODEL)
    # cc_nominal['ion_velocity'] = ui_avg
    # cc_nominal['ion_current_density'] = j_cathode

    # Plot results
    r_grid, alpha_grid = [r.reshape((N, N)), alpha.reshape((N, N))]
    x_grid = r_grid * np.cos(alpha_grid)
    y_grid = r_grid * np.sin(alpha_grid)
    j_ion_grid = j_ion.reshape((N, N))

    # Plot results
    plt.figure()
    c = plt.contourf(x_grid, y_grid, j_ion_grid, 60, cmap='jet')
    cbar = plt.colorbar(c)
    cbar.set_label(r'Ion current density ($A/m^2$)')
    plt.xlabel(r'Distance from thruster exit [m]')
    plt.ylabel(r'Distance from channel centerline [m]')
    plt.tight_layout()
    plt.show()


def main():
    # Run nominal case
    run_nominal()

    # Test cathode coupling model
    # test_cc()


if __name__ == '__main__':
    main()
