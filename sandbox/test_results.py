import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_qoi(ax, x, qoi, xlabel, ylabel, logx=False):
    """ Plot QOI with 5%, 50%, 95% percentiles against x
    x: (Nx)
    qoi: (Nx, Ns)
    """
    p5 = np.percentile(qoi, 5, axis=1)
    med = np.percentile(qoi, 50, axis=1)
    p95 = np.percentile(qoi, 95, axis=1)
    ax.plot(x, med, '-ko', fillstyle='none', label='Model')
    ax.fill_between(x, p5, p95, alpha=0.5, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale('log')


def plot_j_ion(ax, pressure, j_ion, alpha, angle=0):
    idx = np.argmin(np.abs(alpha-angle))
    plot_qoi(ax, pressure, j_ion[:, :, idx], xlabel='Normalized pressure', ylabel=r'Ion current density [$A/m^2$]', logx=False)
    files = os.listdir(Path('../data/DLC-2014/Ion current density'))
    files = sorted(files, key=lambda ele: float(ele.split('_')[2]))
    j_exp = np.zeros(len(files))
    for i, f in enumerate(files):
        data = np.loadtxt(Path('../data/DLC-2014/Ion current density') / f, skiprows=1, delimiter=',')
        idx = np.argmin(np.abs(data[:, 1]*(np.pi/180) - angle))
        j_exp[i] = data[idx, 2]
    ax.plot(pressure[1:], j_exp, 'ro', fillstyle='none', label='Experimental')
    ax.legend()
    ax.set_title(f'Angle={round(angle*180/np.pi)} deg')


def main():
    # Find the most recently generated set of results
    dir = Path('../results')
    files = os.listdir(dir)
    files.remove('.gitignore')
    files.remove('feedforward_mc')
    most_recent = '1900'
    for f in files:
        if f > most_recent:
            most_recent = f

    # List of folders with MC results
    folders = os.listdir(dir / most_recent)
    Nx = len(folders)                                       # Number of pressure locations
    Ns = len(os.listdir(dir / most_recent / folders[0]))    # Number of MC samples
    folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))

    # Allocate space for QoIs
    pressure = np.zeros(Nx)
    v_cc = np.zeros((Nx, Ns))
    ui_avg = np.zeros((Nx, Ns))
    I_B0 = np.zeros((Nx, Ns))
    mass_eff = np.zeros((Nx, Ns))
    voltage_eff = np.zeros((Nx, Ns))
    discharge_current = np.zeros((Nx, Ns))
    thrust = np.zeros((Nx, Ns))
    cathode_density = np.zeros((Nx, Ns))
    Ni = 50
    ion_current_density = np.zeros((Nx, Ns, Ni))

    # Loop and obtain QoIs from json data files
    ns = 0
    for i, folder in enumerate(folders):
        pressure[i] = float(folder.split('_')[0])
        for j, filename in enumerate(os.listdir(dir / most_recent / folders[i])):
            if 'exc' not in str(filename):
                ns += 1
                with open(dir/most_recent/folder/filename, 'r') as fd:
                    data = json.load(fd)
                    v_cc[i, j] = data['model0']['output']['cathode_potential']
                    ui_avg[i, j] = data['model1']['output']['avg_ion_velocity']
                    I_B0[i, j] = data['model1']['output']['I_B0']
                    mass_eff[i, j] = data['model1']['output']['mass_eff'][0]
                    voltage_eff[i, j] = data['model1']['output']['voltage_eff'][0]
                    discharge_current[i, j] = data['model1']['output']['discharge_current'][0]
                    thrust[i, j] = data['model1']['output']['thrust'][0]
                    cathode_density[i, j] = data['model2']['output']['cathode_current_density']
                    ion_current_density[i, j, :] = np.atleast_1d(data['model2']['output']['ion_current_density'])
                    r = np.atleast_1d(data['model2']['output']['r'])
                    alpha = np.atleast_1d(data['model2']['output']['alpha'])

    # Normalize pressure:
    pressure = pressure / 1e-5

    # Plot performance
    fig, axs = plt.subplots(2, 2)
    plot_qoi(axs[0, 0], pressure, thrust*1000, xlabel='Normalized pressure', ylabel='Thrust [mN]')
    plot_qoi(axs[0, 1], pressure, discharge_current, xlabel='Normalized pressure', ylabel='Discharge current [A]')
    plot_qoi(axs[1, 0], pressure, mass_eff, xlabel='Normalized pressure', ylabel='Mass efficiency')
    plot_qoi(axs[1, 1], pressure, voltage_eff, xlabel='Normalized pressure', ylabel='Voltage efficiency')
    fig.set_size_inches(7, 7)
    fig.tight_layout(pad=1.0)
    plt.show()

    # Plot intermediate QoIs
    fig, axs = plt.subplots(2, 2)
    plot_qoi(axs[0, 0], pressure, v_cc, xlabel='Normalized pressure', ylabel=r'$V_{cc}$ [V]')
    plot_qoi(axs[0, 1], pressure, ui_avg, xlabel='Normalized pressure', ylabel='Exit ion velocity [m/s]')
    plot_qoi(axs[1, 0], pressure, I_B0, xlabel='Normalized pressure', ylabel='Exit current [A]')
    plot_qoi(axs[1, 1], pressure, cathode_density, xlabel='Normalized pressure', ylabel=r'Cathode current density [$A/m^2$]')
    fig.set_size_inches(7, 7)
    fig.tight_layout(pad=1.0)
    plt.show()

    # Compare to Diamant 2014 j_ion at r=1m, various angles
    fig, axs = plt.subplots(2, 2)
    plot_j_ion(axs[0, 0], pressure, ion_current_density, alpha, angle=0)
    plot_j_ion(axs[0, 1], pressure, ion_current_density, alpha, angle=30*np.pi/180)
    plot_j_ion(axs[1, 0], pressure, ion_current_density, alpha, angle=60*np.pi/180)
    plot_j_ion(axs[1, 1], pressure, ion_current_density, alpha, angle=90*np.pi/180)
    fig.set_size_inches(7, 7)
    fig.tight_layout(pad=1.0)
    plt.show()

    # Plot ion current density average
    # j_ion = np.sum(ion_current_density, axis=0) / ns
    # r_grid, alpha_grid = [r.reshape((N, N)), alpha.reshape((N, N))]
    # x_grid = r_grid * np.cos(alpha_grid)
    # y_grid = r_grid * np.sin(alpha_grid)
    # j_ion_grid = j_ion.reshape((N, N))

    # Plot results
    # plt.figure()
    # c = plt.contourf(x_grid, y_grid, j_ion_grid, 60, cmap='jet')
    # cbar = plt.colorbar(c)
    # cbar.set_label(r'Average ion current density ($A/m^2$)')
    # plt.xlabel(r'Distance from thruster exit [m]')
    # plt.ylabel(r'Distance from channel centerline [m]')
    # plt.tight_layout()
    # plt.show()

    # Plot histograms of QoIs
    # fig, axs = plt.subplots(2, 2)
    # bins=10
    # axs[0, 0].hist(v_cc, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[0, 0].set_xlabel(r'$V_{cc}$ [V]')
    # axs[0, 1].hist(ui_avg, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[0, 1].set_xlabel(r'$\bar{u}_i$ [m/s]')
    # axs[1, 0].hist(I_B0, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[1, 0].set_xlabel(r'Total current $I_{B,0}$ [A]')
    # axs[1, 1].hist(cathode_density, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[1, 1].set_xlabel(r'Cathode coupling $j$ [$A/m^2$]')
    # fig.set_size_inches(7, 7)
    # fig.tight_layout(pad=1.0)
    # plt.show()


if __name__ == '__main__':
    main()
