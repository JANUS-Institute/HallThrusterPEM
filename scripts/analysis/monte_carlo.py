"""Script for surrogate-enabled Monte Carlo forward UQ analysis."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from amisc.system import SystemSurrogate

from hallmd.data.loader import spt100_data
from hallmd.models.plume import jion_reconstruct
from hallmd.models.thruster import uion_reconstruct
from hallmd.utils import plot_qoi

PROJECT_ROOT = Path('../..')


def spt100_monte_carlo(Ns=100, surr_dir='mf_2023-11-02T17.57.44'):
    """Plot `[V_cc, T, uion, jion]` against SPT-100 experimental data with UQ bounds."""
    exp_data = spt100_data()
    model = SystemSurrogate.load_from_file(PROJECT_ROOT / 'results' / surr_dir / 'multi-fidelity'
                                           / 'sys' / 'sys_final.pkl')
    # Cathode coupling voltage
    data = exp_data['V_cc'][0]
    pb = data['x'][:, 0]
    idx = np.argsort(pb)
    Nx = data['x'].shape[0]
    xs = np.empty((Nx, Ns, len(model.exo_vars)))
    for i in range(Nx):
        nominal = {'PB': pb[i], 'Va': data['x'][i, 1], 'mdot_a': data['x'][i, 2]}
        xs[i, :, :] = model.sample_inputs((Ns,), use_pdf=True, nominal=nominal)
    ys = np.squeeze(model(xs, qois=['V_cc'], training=True), axis=-1)
    fig, ax = plt.subplots()
    yerr = np.squeeze(2 * np.sqrt(data['var_y']))
    ax.errorbar(10 ** pb, np.squeeze(data['y']), yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                label='Experiment', markersize=4)
    plot_qoi(ax, 10 ** pb[idx], ys[idx, :], 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=True)
    fig.tight_layout()
    plt.show()

    # Thrust
    data = exp_data['T'][0]
    pb = data['x'][:, 0]
    idx = np.argsort(pb)
    Nx = data['x'].shape[0]
    xs = np.empty((Nx, Ns, len(model.exo_vars)))
    for i in range(Nx):
        nominal = {'PB': pb[i], 'Va': data['x'][i, 1], 'mdot_a': data['x'][i, 2]}
        xs[i, :, :] = model.sample_inputs((Ns,), use_pdf=True, nominal=nominal)
    ys = np.squeeze(model(xs, qois=['T'], training=True), axis=-1)
    fig, ax = plt.subplots()
    yerr = np.squeeze(2 * np.sqrt(data['var_y']))
    ax.errorbar(10**pb, np.squeeze(data['y'])*1000, yerr=yerr*1000, fmt='or', capsize=3, markerfacecolor='none',
                label='Experiment', markersize=4)
    plot_qoi(ax, 10**pb[idx], ys[idx, :]*1000, 'Background pressure (Torr)', 'Thrust (mN)', legend=True)
    fig.tight_layout()
    plt.show()

    # Ion velocity
    data = exp_data['uion'][0]
    pb = data['x'][:, 0]
    Nx = data['x'].shape[0]
    xs = np.empty((Nx, Ns, len(model.exo_vars)))
    for i in range(Nx):
        nominal = {'PB': pb[i], 'Va': data['x'][i, 1], 'mdot_a': data['x'][i, 2]}
        xs[i, :, :] = model.sample_inputs((Ns,), use_pdf=True, nominal=nominal)
    qois = [var for var in model.coupling_vars if str(var).startswith('uion')]
    ys = model(xs, qois=qois, training=True)
    zg, uion_g = uion_reconstruct(ys)

    fig, ax = plt.subplots(1, Nx, sharey='row')
    for i in range(Nx):
        yerr = 2 * np.sqrt(data['var_y'][i, :, 0])
        ax[i].errorbar(data['loc'][i, :, 0]*1000, data['y'][i, :, 0], yerr=yerr, fmt='or', capsize=3,
                       markerfacecolor='none', label='Experiment', markersize=4)
        ylabel = 'Axial ion velocity (m/s)' if i == 0 else ''
        legend = i == Nx-1
        plot_qoi(ax[i], zg*1000, uion_g[i, :, :].T, 'Axial distance from anode (mm)', ylabel, legend=legend)
        ax[i].set_title(f'{10**pb[i]:.2E} Torr')
    fig.set_size_inches(Nx*3, 3)
    fig.tight_layout()
    plt.show()

    # Ion current density
    data = exp_data['jion'][0]
    pb = data['x'][:, 0]
    Nx = data['x'].shape[0]
    xs = np.empty((Nx, Ns, len(model.exo_vars)))
    for i in range(Nx):
        nominal = {'PB': pb[i], 'Va': data['x'][i, 1], 'mdot_a': data['x'][i, 2], 'r_m': data['loc'][i, 0, 0]}
        xs[i, :, :] = model.sample_inputs((Ns,), use_pdf=True, nominal=nominal)
    qois = [var for var in model.coupling_vars if str(var).startswith('jion')]
    ys = model(xs, qois=qois, training=True)
    alpha_g = data['loc'][0, :, 1]
    keep_idx = np.where(np.abs(alpha_g) < np.pi/2)[0]
    alpha_g = alpha_g[keep_idx]
    alpha_i, jion_i = jion_reconstruct(ys, alpha=alpha_g)

    fig, ax = plt.subplots(2, int(Nx/2), sharey='row', sharex='col')
    idx = 0
    for i in range(2):
        for j in range(int(Nx/2)):
            yerr = 2 * np.sqrt(data['var_y'][idx, keep_idx, 0])
            ax[i, j].errorbar(alpha_g*180/np.pi, data['y'][idx, keep_idx, 0], yerr=yerr, fmt='or', capsize=3,
                              markerfacecolor='none', label='Experiment', markersize=4)
            xlabel = 'Angle from thruster centerline (deg)' if i == 1 else ''
            ylabel = 'Ion current density ($A/m^2$)' if j == 0 else ''
            legend = i == 0 and j == int(Nx/2) - 1
            plot_qoi(ax[i, j], alpha_i*180/np.pi, jion_i[idx, :, :].T, xlabel, ylabel, legend=legend)
            ax[i, j].set_title(f'{10**pb[idx]:.2E} Torr')
            ax[i, j].set_yscale('log')
            idx += 1
    fig.set_size_inches(3*int(Nx/2), 6)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    spt100_monte_carlo()
