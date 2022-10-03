import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    dir = Path('../results/feedforward_mc')
    files = os.listdir(dir)
    v_cc = []
    ui_avg = []
    I_B0 = []
    cathode_density = []
    N = 50
    ion_current_density = np.zeros((len(files), N*N))
    ns = 0
    for i, f in enumerate(files):
        if 'exc' not in str(f):
            ns += 1
            with open(dir/str(f), 'r') as fd:
                data = json.load(fd)
                v_cc.append(data['model0']['output']['cathode_potential'])
                ui_avg.append(data['model1']['output']['avg_ion_velocity'])
                I_B0.append(data['model1']['output']['I_B0'])
                cathode_density.append(data['model2']['output']['cathode_current_density'])
                ion_current_density[i, :] = np.atleast_1d(data['model2']['output']['ion_current_density'])
                r = np.atleast_1d(data['model2']['output']['r'])
                alpha = np.atleast_1d(data['model2']['output']['alpha'])

    # Plot ion current density average
    j_ion = np.sum(ion_current_density, axis=0) / ns
    r_grid, alpha_grid = [r.reshape((N, N)), alpha.reshape((N, N))]
    x_grid = r_grid * np.cos(alpha_grid)
    y_grid = r_grid * np.sin(alpha_grid)
    j_ion_grid = j_ion.reshape((N, N))

    # Plot results
    plt.figure()
    c = plt.contourf(x_grid, y_grid, j_ion_grid, 60, cmap='jet')
    cbar = plt.colorbar(c)
    cbar.set_label(r'Average ion current density ($A/m^2$)')
    plt.xlabel(r'Distance from thruster exit [m]')
    plt.ylabel(r'Distance from channel centerline [m]')
    plt.tight_layout()
    plt.show()

    # Plot histograms of QoIs
    fig, axs = plt.subplots(2, 2)
    bins=10
    axs[0, 0].hist(v_cc, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[0, 0].set_xlabel(r'$V_{cc}$ [V]')
    axs[0, 1].hist(ui_avg, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[0, 1].set_xlabel(r'$\bar{u}_i$ [m/s]')
    axs[1, 0].hist(I_B0, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[1, 0].set_xlabel(r'Total current $I_{B,0}$ [A]')
    axs[1, 1].hist(cathode_density, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[1, 1].set_xlabel(r'Cathode coupling $j$ [$A/m^2$]')
    fig.set_size_inches(7, 7)
    fig.tight_layout(pad=1.0)
    plt.show()


if __name__ == '__main__':
    main()
