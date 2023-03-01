# Standard imports
import matplotlib.pyplot as plt
import sys
import numpy as np
import logging
from pathlib import Path
import os

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.pem import feedforward_pem
from models.cc import cathode_coupling_model_feedforward as cc_model
from utils import parse_input_file, data_write, data_load, ax_default, set_model_inputs


def test_ui_nominal():
    # Loop over each pressure in ui dataset
    ui_data = np.loadtxt(Path('../data/spt100/ui_dataset5.csv'), delimiter=',', skiprows=1)
    Np = 3
    pb = ui_data[:, 4].reshape((Np, -1))
    z = ui_data[:, 5].reshape((Np, -1))
    ui = ui_data[:, 6].reshape((Np, -1))
    ui_var = ui_data[:, 7].reshape((Np, -1))

    for pressure in pb[:, 0]:
        # Set system inputs
        set_model_inputs('system', {'background_pressure_Torr': float(pressure)})

        # Load nominal conditions
        cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
        thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
        plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')
        model_inputs = [cc_nominal, thruster_nominal, plume_nominal]

        # Run feedforward pem
        pem_output = feedforward_pem(model_inputs)
        data_write(pem_output, f'pem_nominal_{pressure:.02}_torr.json', dir='../results/nominal/ui')


def view_ui_nominal():
    """Compare nominal model predictions to data"""
    ui_data = np.loadtxt(Path('../data/spt100/ui_dataset5.csv'), delimiter=',', skiprows=1)
    Np = 3
    pb = ui_data[:, 4].reshape((Np, -1))
    z = ui_data[:, 5].reshape((Np, -1))
    ui = ui_data[:, 6].reshape((Np, -1))
    ui_var = ui_data[:, 7].reshape((Np, -1))

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(11, 4)

    for i, pressure in enumerate(pb[:, 0]):
        # Load pem results
        pem_results = data_load(f'pem_nominal_{pressure:.02}_torr.json', dir='../results/nominal/ui')
        z_pem = np.atleast_1d(pem_results['thruster']['output']['z'])
        ui_pem = np.atleast_1d(pem_results['thruster']['output']['ui_1'])

        # Plot model against experimental data
        yerr = 2 * np.sqrt(ui_var[i, :])
        ax[i].errorbar(z[i, :] * 1000, ui[i, :], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                       label='Experimental', markersize=5)
        ax[i].plot(z_pem * 1000, ui_pem, '-k', label='PEM')
        ax[i].set_title(f'{pressure:.02} Torr')
        ax_default(ax[i], 'Axial distance from anode (mm)', 'Average ion axial velocity ($m/s$)')

    fig.tight_layout()
    plt.show()


def test_cc_nominal():
    # Loop over each operating condition in vcc dataset
    cc_data = np.loadtxt(Path('../data/spt100/vcc_dataset8.csv'), delimiter=',', skiprows=1)
    pb = cc_data[:, 4]
    vcc = cc_data[:, 5]
    vcc_var = cc_data[:, 6]
    vcc_model = np.zeros(vcc.shape[0])

    for i, pressure in enumerate(pb):
        # Set system inputs
        set_model_inputs('system', {'background_pressure_Torr': float(pressure)})

        # Load nominal conditions
        cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')

        # Run cc_model
        cc_output = cc_model(cc_nominal)
        vcc_model[i] = cc_output['cathode_potential']

    idx = np.argsort(pb)
    fig, ax = plt.subplots()
    yerr = 2 * np.sqrt(vcc_var)
    ax.errorbar(pb, vcc, yerr=yerr, fmt='or', capsize=3, markerfacecolor='none', label='Experimental', markersize=5)
    ax.plot(pb[idx], vcc_model[idx], '-k', label='PEM')
    ax.set_ylim([30, 34])
    ax_default(ax, 'Background pressure (Torr)', 'Cathode coupling voltage (V)')
    fig.tight_layout()
    plt.show()


def test_jion_nominal():
    # Loop over each pressure
    jion_data = np.loadtxt(Path('../data/spt100/jion_dataset4.csv'), delimiter=',', skiprows=1)
    Np = 8
    mdot_a = jion_data[:, 2].reshape((Np, -1)) * 1e-6
    pb = jion_data[:, 4].reshape((Np, -1))
    r_m = jion_data[:, 5].reshape((Np, -1))
    alpha_deg = jion_data[:, 6].reshape((Np, -1))
    jion = jion_data[:, 7].reshape((Np, -1))        # mA/cm^2
    jion_var = jion_data[:, 8].reshape((Np, -1))

    for i in range(Np):
        # Set inputs
        set_model_inputs('system', {'background_pressure_Torr': float(pb[i, 0])})
        set_model_inputs('thruster', {'anode_mass_flow_rate': float(mdot_a[i, 0])})
        set_model_inputs('plume', {'r_m': list(r_m[i, :]), 'alpha_deg': list(alpha_deg[i, :])})

        # Load inputs (at nominal)
        cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
        thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
        plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')
        model_inputs = [cc_nominal, thruster_nominal, plume_nominal]

        # Run feedforward pem
        pem_output = feedforward_pem(model_inputs)
        data_write(pem_output, f'pem_nominal_{pb[i, 0]:.02}_torr.json', dir='../results/nominal/jion')


def view_jion_nominal():
    # Loop over each pressure
    jion_data = np.loadtxt(Path('../data/spt100/jion_dataset4.csv'), delimiter=',', skiprows=1)
    Np = 8
    mdot_a = jion_data[:, 2].reshape((Np, -1)) * 1e-6
    pb = jion_data[:, 4].reshape((Np, -1))
    r_m = jion_data[:, 5].reshape((Np, -1))
    alpha_deg = jion_data[:, 6].reshape((Np, -1))
    jion = jion_data[:, 7].reshape((Np, -1))  # mA/cm^2
    jion_var = jion_data[:, 8].reshape((Np, -1))

    fig, ax = plt.subplots(4, 2, sharex='col')
    fig.set_size_inches(6, 12)

    idx = 0
    for j in range(2):
        for i in range(4):
            # Load pem results
            pem_results = data_load(f'pem_nominal_{pb[idx, 0]:.02}_torr.json', dir='../results/nominal/jion')
            jion_pem = np.atleast_1d(pem_results['plume']['output']['ion_current_density']) * (1/1000) * (100**2)

            # Plot model against experimental data
            yerr = 2 * np.sqrt(jion_var[idx, :])
            ax[i, j].errorbar(alpha_deg[idx, :], jion[idx, :], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                           label='Experimental', markersize=5)
            # ax[i, j].plot(alpha_deg[idx, :], jion_pem, '-k', label='PEM')
            ax[i, j].set_title(f'{pb[idx, 0]:.02} Torr')
            ax_default(ax[i, j], 'Angle from thruster centerline (deg)', 'Ion current density ($mA/cm^2$)')
            idx += 1

    fig.tight_layout()
    plt.show()


def test_thrust_nominal():
    # Loop over and collect thrust data
    files = [f for f in os.listdir(Path('../data/spt100')) if f.startswith('thrust')]
    thrust_data = np.zeros((1, 7))
    for fname in files:
        data = np.loadtxt(Path(f'../data/spt100/{fname}'), delimiter=',', skiprows=1)
        data = np.atleast_2d(data).reshape((-1, 7))
        thrust_data = np.concatenate((thrust_data, data), axis=0)
    thrust_data = thrust_data[1:, :]

    # Run model at each operating condition of the data
    Nd = thrust_data.shape[0]
    thrust_pem = np.zeros(Nd)
    for i in range(Nd):
        # Operating conditions
        Va = thrust_data[i, 1]              # Anode voltage (V)
        mdot_a = thrust_data[i, 2] * 1e-6   # Mass flow rate at anode (mg/s)
        TB = thrust_data[i, 3]              # Background temperature (K)
        PB = thrust_data[i, 4]              # Background pressure (Torr)

        # Set inputs
        set_model_inputs('system', {'background_pressure_Torr': float(PB), 'background_temperature_K': float(TB),
                                    'anode_potential': float(Va)})
        set_model_inputs('thruster', {'anode_mass_flow_rate': float(mdot_a)})

        # Load inputs (at nominal)
        cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
        thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
        plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')
        model_inputs = [cc_nominal, thruster_nominal, plume_nominal]

        # Run feedforward pem
        pem_output = feedforward_pem(model_inputs)
        data_write(pem_output, f'pem_nominal_{i}.json', dir='../results/nominal/thrust')


def view_thrust_nominal():
    # Loop over and collect thrust data
    files = [f for f in os.listdir(Path('../data/spt100')) if f.startswith('thrust')]
    thrust_data = np.zeros((1, 7))
    for fname in files:
        data = np.loadtxt(Path(f'../data/spt100/{fname}'), delimiter=',', skiprows=1)
        data = np.atleast_2d(data).reshape((-1, 7))
        thrust_data = np.concatenate((thrust_data, data), axis=0)
    thrust_data = thrust_data[1:, :]
    thrust_exp = thrust_data[:, 5]      # mN
    thrust_var = thrust_data[:, 6]      # mN^2
    Nd = thrust_exp.shape[0]

    # Loop over each result
    thrust_pem = np.zeros(Nd)
    sort_func = lambda ele: int(ele.split('_')[2].split('.')[0])
    files = os.listdir(Path('../results/nominal/thrust'))
    files.sort(key=sort_func)
    for i, fname in enumerate(files):
        pem_results = data_load(fname, dir='../results/nominal/thrust')
        thrust_pem[i] = np.atleast_1d(pem_results['thruster']['output']['thrust']) * 1000  # mN

    # Plot results
    fig, ax = plt.subplots()
    xerr = 2 * np.sqrt(thrust_var)
    x = np.linspace(thrust_exp.min(), thrust_exp.max(), 100)
    ax.errorbar(thrust_exp, thrust_pem, xerr=xerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=5)
    ax.plot(x, x, '--k')
    ax_default(ax, 'Actual thrust (mN)', 'Predicted thrust (mN)', legend=False)
    fig.tight_layout()
    plt.show()

    # Plot error
    pct_error = ((thrust_pem - thrust_exp) / thrust_exp) * 100
    fig, ax = plt.subplots()
    ax.hist(pct_error, density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
    ax_default(ax, r'Thrust percent error (%)', '', legend=False)
    fig.tight_layout()
    plt.show()


def main():
    # Run nominal case
    # test_ui_nominal()
    # view_ui_nominal()
    # test_cc_nominal()
    # test_jion_nominal()
    # view_jion_nominal()
    # test_thrust_nominal()
    view_thrust_nominal()


if __name__ == '__main__':
    main()
