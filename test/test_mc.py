# Standard imports
import sys
import numpy as np
import logging
import copy
from joblib import Parallel, delayed, cpu_count
import datetime
from datetime import timezone
import os
from pathlib import Path
import math
import time
import traceback
import json
import matplotlib.pyplot as plt

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.pem import feedforward_pem
from models.cc import cathode_coupling_model_feedforward as cc_model
from utils import parse_input_file, data_write, ModelRunException, data_load, ax_default, spt100_data


def plot_qoi(ax, x, qoi, xlabel, ylabel, logx=False, legend=False, linestyle='-ko'):
    """ Plot QOI with 5%, 50%, 95% percentiles against x
    x: (Nx)
    qoi: (Nx, Ns)
    """
    p5 = np.percentile(qoi, 5, axis=1)
    med = np.percentile(qoi, 50, axis=1)
    p95 = np.percentile(qoi, 95, axis=1)
    ax.plot(x, med, linestyle, fillstyle='none', label='Model')
    ax.fill_between(x, p5, p95, alpha=0.5, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    ax_default(ax, xlabel, ylabel, legend=legend)
    if logx:
        ax.set_xscale('log')


def plot_mc_experimental(base_path):
    """Plot thrust, ion velocity, and ion current density against experimental data with UQ bounds"""
    # Thrust
    folders = os.listdir(base_path / 'thrust')
    Nx = len(folders)  # Number of pressure locations
    Ns = len(os.listdir(base_path / 'thrust' / folders[0]))  # Number of MC samples
    folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))
    pb = np.zeros(Nx)
    pred = np.zeros((Nx, Ns))

    # Load experimental data
    data = spt100_data(exclude=['vcc', 'ion_velocity', 'ion_current_density'])
    exp = np.zeros((Nx, 3))
    for i, data_tuple in enumerate(data):
        exp[i, 0] = data_tuple[1]['background_pressure_Torr']
        exp[i, 1] = data_tuple[2]
        exp[i, 2] = ((data_tuple[2] * data_tuple[3]['value']/100) / 2) ** 2

    # Load predictions
    for i, folder in enumerate(folders):
        pb[i] = float(folder.split('_')[0])
        for j, filename in enumerate(os.listdir(base_path / 'thrust' / folder)):
            if 'exc' not in str(filename):
                with open(base_path / 'thrust' / folder / filename, 'r') as fd:
                    data = json.load(fd)
                    pred[i, j] = data['thruster']['output']['thrust'][0]

    # Plot thrust
    fig, ax = plt.subplots()
    yerr = 2 * np.sqrt(exp[:, 2])
    ax.errorbar(exp[:, 0], exp[:, 1]*1000, yerr=yerr*1000, fmt='or', capsize=3, markerfacecolor='none',
                label='Experimental', markersize=5)
    plot_qoi(ax, pb, pred*1000, 'Background pressure [torr]', 'Thrust [mN]', legend=True)
    fig.tight_layout()
    plt.show()

    # Ion velocity
    folders = os.listdir(base_path / 'ion_velocity')
    Nx = len(folders)  # Number of pressure locations
    Ns = len(os.listdir(base_path / 'ion_velocity' / folders[0]))  # Number of MC samples
    folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))
    pb = np.zeros(Nx)
    n_grid = 152
    z_m = np.zeros(n_grid)
    pred = np.zeros((Nx, n_grid, Ns))

    # Load experimental data
    data = spt100_data(exclude=['vcc', 'thrust', 'ion_current_density'])
    data = sorted(data, key=lambda ele: ele[1]['background_pressure_Torr'])
    ndata = len(data[0][1]['z_m'])
    exp = np.zeros((Nx, ndata, 3))
    for i, data_tuple in enumerate(data):
        exp[i, :, 0] = np.atleast_1d(data_tuple[1]['z_m'])
        exp[i, :, 1] = np.atleast_1d(data_tuple[2])
        exp[i, :, 2] = data_tuple[3]['value']

    # Load predictions
    for i, folder in enumerate(folders):
        pb[i] = float(folder.split('_')[0])
        for j, filename in enumerate(os.listdir(base_path / 'ion_velocity' / folder)):
            if 'exc' not in str(filename):
                with open(base_path / 'ion_velocity' / folder / filename, 'r') as fd:
                    data = json.load(fd)
                    z_m[:] = np.atleast_1d(data['thruster']['output']['z'])
                    pred[i, :, j] = np.atleast_1d(data['thruster']['output']['ui_1'])

    # Plot ion velocity
    fig, ax = plt.subplots(1, Nx, sharey='row')
    for i in range(Nx):
        yerr = 2 * np.sqrt(exp[i, :, 2])
        ax[i].errorbar(exp[i, :, 0]*1000, exp[i, :, 1], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                       label='Experimental', markersize=5)
        ylabel = 'Axial ion velocity [m/s]' if i == 0 else ''
        legend = i == 2
        plot_qoi(ax[i], z_m*1000, pred[i, :, :], 'Axial distance from anode [mm]', ylabel,
                 legend=legend, linestyle='-k')
        ax[i].set_title(f'{pb[i]} torr')
    fig.set_size_inches(9, 3)
    fig.tight_layout()
    plt.show()

    # Ion current density
    folders = os.listdir(base_path / 'ion_current_density')
    Nx = len(folders)  # Number of pressure locations
    Ns = len(os.listdir(base_path / 'ion_current_density' / folders[0]))  # Number of MC samples
    folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))
    pb = np.zeros(Nx)

    # Load experimental data
    data = spt100_data(exclude=['vcc', 'thrust', 'ion_velocity'])
    data = sorted(data, key=lambda ele: ele[1]['background_pressure_Torr'])
    ndata = len(data[0][1]['alpha_deg'])
    exp = np.zeros((Nx, ndata, 3))
    alpha_deg = np.zeros(ndata)
    pred = np.zeros((Nx, ndata, Ns))
    for i, data_tuple in enumerate(data):
        exp[i, :, 0] = np.atleast_1d(data_tuple[1]['alpha_deg'])
        exp[i, :, 1] = np.atleast_1d(data_tuple[2])
        exp[i, :, 2] = ((np.atleast_1d(data_tuple[2]) * data_tuple[3]['value']/100) / 2) ** 2

    # Load predictions
    for i, folder in enumerate(folders):
        pb[i] = float(folder.split('_')[0])
        for j, filename in enumerate(os.listdir(base_path / 'ion_current_density' / folder)):
            if 'exc' not in str(filename):
                with open(base_path / 'ion_current_density' / folder / filename, 'r') as fd:
                    data = json.load(fd)
                    alpha_deg[:] = np.atleast_1d(data['plume']['input']['alpha_deg'])
                    pred[i, :, j] = np.atleast_1d(data['plume']['output']['ion_current_density'])

    # Plot ion velocity
    fig, ax = plt.subplots(2, int(Nx/2), sharey='row', sharex='col')
    idx = 0
    for i in range(2):
        for j in range(int(Nx/2)):
            yerr = 2 * np.sqrt(exp[idx, :, 2])
            ax[i, j].errorbar(exp[idx, :, 0], exp[idx, :, 1]*0.1, yerr=yerr*0.1, fmt='or', capsize=3,
                              markerfacecolor='none', label='Experimental', markersize=5)
            xlabel = 'Angle from thruster centerline [deg]' if i == 1 else ''
            ylabel = 'Ion current density [$mA/cm^2$]' if j == 0 else ''
            legend = i == 0 and j == int(Nx/2) - 1
            plot_qoi(ax[i, j], alpha_deg, pred[idx, :, :]*0.1, xlabel, ylabel, legend=legend, linestyle='-k')
            ax[i, j].set_title(f'{pb[idx]} torr')
            ax[i, j].set_yscale('log')
            idx += 1
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    plt.show()


def plot_mc(base_path):
    """Plot all QoIs and histograms from an MC run"""
    # List of folders with MC results
    folders = os.listdir(base_path)
    Nx = len(folders)  # Number of pressure locations
    Ns = len(os.listdir(base_path / folders[0]))  # Number of MC samples
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
    # cathode_density = np.zeros((Nx, Ns))
    # Ni = 50
    # ion_current_density = np.zeros((Nx, Ns, Ni))

    # Loop and obtain QoIs from json data files
    ns = 0
    for i, folder in enumerate(folders):
        pressure[i] = float(folder.split('_')[0])
        for j, filename in enumerate(os.listdir(base_path / folder)):
            if 'exc' not in str(filename):
                ns += 1
                with open(base_path / folder / filename, 'r') as fd:
                    data = json.load(fd)
                    v_cc[i, j] = data['cc']['output']['cathode_potential']
                    ui_avg[i, j] = data['thruster']['output']['avg_ion_velocity']
                    I_B0[i, j] = data['thruster']['output']['I_B0']
                    mass_eff[i, j] = data['thruster']['output']['mass_eff'][0]
                    voltage_eff[i, j] = data['thruster']['output']['voltage_eff'][0]
                    discharge_current[i, j] = data['thruster']['output']['discharge_current'][0]
                    thrust[i, j] = data['thruster']['output']['thrust'][0]
                    # cathode_density[i, j] = data['model2']['output']['cathode_current_density']
                    # ion_current_density[i, j, :] = np.atleast_1d(data['model2']['output']['ion_current_density'])
                    # r = np.atleast_1d(data['model2']['output']['r'])
                    # alpha = np.atleast_1d(data['model2']['output']['alpha'])

    # Plot performance
    fig, axs = plt.subplots(3, 2)
    plot_qoi(axs[0, 0], pressure, thrust * 1000, xlabel='Pressure [torr]', ylabel='Thrust [mN]')
    axs[0, 0].set_ylim([0, 105])
    plot_qoi(axs[0, 1], pressure, discharge_current, xlabel='Pressure [torr]', ylabel='Discharge current [A]')
    axs[0, 1].set_ylim([0, 65])
    plot_qoi(axs[1, 0], pressure, mass_eff, xlabel='Pressure [torr]', ylabel='Mass efficiency')
    axs[1, 0].set_ylim([0, 1.5])
    plot_qoi(axs[1, 1], pressure, voltage_eff, xlabel='Pressure [torr]', ylabel='Voltage efficiency')
    axs[1, 1].set_ylim([0, 1.5])
    plot_qoi(axs[2, 0], pressure, v_cc, xlabel='Pressure [torr]', ylabel=r'Cathode coupling voltage [V]')
    axs[2, 0].set_ylim([30, 34])
    plot_qoi(axs[2, 1], pressure, I_B0, xlabel='Pressure [torr]', ylabel='Total beam current [A]')
    axs[2, 1].set_ylim([0, 10])
    # plot_qoi(axs[1, 0], pressure, ui_avg, xlabel='Normalized pressure', ylabel='Exit ion velocity [m/s]')
    # plot_qoi(axs[1, 1], pressure, cathode_density, xlabel='Normalized pressure',
    #          ylabel=r'Cathode current density [$A/m^2$]')
    fig.set_size_inches(7, 10.5)
    fig.tight_layout(pad=1.0)
    plt.show()

    # Plot histograms of QoIs
    bins = 10
    idx = 4
    print(pressure[idx])
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].hist(thrust[idx, :] * 1000, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[0, 0].set_xlabel(r'Thrust [mN]')
    axs[0, 0].set_xlim([0, 105])
    axs[0, 1].hist(discharge_current[idx, :], density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[0, 1].set_xlabel(r'Discharge current [A]')
    axs[0, 1].set_xlim([0, 65])
    axs[1, 0].hist(mass_eff[idx, :], density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[1, 0].set_xlabel(r'Mass efficiency')
    axs[1, 0].set_xlim([0, 1.5])
    axs[1, 1].hist(voltage_eff[idx, :], density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[1, 1].set_xlabel(r'Voltage efficiency')
    axs[1, 1].set_xlim([0, 1.5])
    axs[2, 0].hist(v_cc[idx, :], density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[2, 0].set_xlabel(r'Cathode coupling voltage [V]')
    axs[2, 0].set_xlim([30, 34])
    axs[2, 1].hist(I_B0[idx, :], density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    axs[2, 1].set_xlabel(r'Total beam current [A]')
    axs[2, 1].set_xlim([0, 10])
    # axs[0, 1].hist(ui_avg, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[0, 1].set_xlabel(r'$\bar{u}_i$ [m/s]')
    # axs[1, 1].hist(cathode_density, density=True, bins=bins, color='r', edgecolor='black', linewidth=1.2)
    # axs[1, 1].set_xlabel(r'Cathode coupling $j$ [$A/m^2$]')
    fig.set_size_inches(7, 10.5)
    fig.tight_layout(pad=1.0)
    plt.show()


def input_sampler(nominal_list, uncertainty_list):
    """Sample all inputs from uncertainty models"""
    model_inputs = list()
    sample_tracker = {}  # keep track of samples that need to be re-used between models
    for i in range(len(nominal_list)):
        nominal_dict = nominal_list[i]
        uncertainty_dict = uncertainty_list[i]
        input_dict = copy.deepcopy(nominal_dict)

        for param, uq_dict in uncertainty_dict.items():
            # Only sample shared parameters once
            if param in sample_tracker:
                input_dict[param] = sample_tracker.get(param)
                continue

            # Sample from uncertainty bounds
            if uq_dict['uncertainty'] == 'uniform_bds':
                lb, ub = uq_dict['value']
                input_dict[param] = np.random.rand() * (ub-lb) + lb
            elif uq_dict['uncertainty'] == 'uniform_pct':
                lb = nominal_dict[param] * (1 - uq_dict['value'])
                ub = nominal_dict[param] * (1 + uq_dict['value'])
                input_dict[param] = np.random.rand() * (ub-lb) + lb
            elif uq_dict['uncertainty'] == 'uniform_tol':
                lb = nominal_dict[param] - uq_dict['value']
                ub = nominal_dict[param] + uq_dict['value']
                input_dict[param] = np.random.rand() * (ub-lb) + lb
            elif uq_dict['uncertainty'] == 'lognormal':
                scale = 1 / np.log10(np.e)
                mean = scale * uq_dict['value'][0]
                var = scale ** 2 * uq_dict['value'][1]
                input_dict[param] = float(np.random.lognormal(mean=mean, sigma=np.sqrt(var), size=1))
            elif uq_dict['uncertainty'] == 'normal':
                input_dict[param] = np.random.randn() * np.sqrt(uq_dict['value'][1]) + uq_dict['value'][0]
            else:
                raise NotImplementedError

            # Update sample tracker
            sample_tracker[param] = input_dict.get(param)

        model_inputs.append(input_dict)

    return model_inputs


def set_inputs(input_dict, inputs_to_set):
    """Set model inputs"""
    for input_name, value in inputs_to_set.items():
        input_dict[input_name] = value


def test_cc_forward(num_samples, n_jobs=-1):
    """Run Monte Carlo on vcc model and compare to data"""
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json', exclude=['parameters'])
    uncertainty_list = [cc_uncertainty]

    # Load experimental data
    exp_data = list()
    data = np.loadtxt(Path('../data/spt100/vcc_dataset8.csv'), delimiter=',', skiprows=1)
    for i in range(data.shape[0]):
        x = {'anode_potential': data[i, 1],
             'anode_mass_flow_rate': data[i, 2] * 1e-6,
             'background_pressure_Torr': data[i, 3]}
        y = data[i, 4]
        var = {'noise_var': 'constant', 'value': (0.3 / 2) ** 2}
        exp_data.append(('cc_voltage', x.copy(), y, var.copy()))

    # Allocate space for predictions
    Nd = len(exp_data)
    fname = 'temp.dat'
    vcc_pred = np.memmap(fname, dtype='float64', shape=(Nd, num_samples), mode='w+')

    # Batch function to run in parallel
    def run_batch(data_idx, sample_idx, vcc):
        # Set operating conditions
        cc_input = cc_nominal.copy()
        operating_condition = exp_data[data_idx][1]
        set_inputs(cc_input, operating_condition)
        nominal_list = [cc_input]

        # Sample inputs and run model
        model_inputs = input_sampler(nominal_list, uncertainty_list)
        pem_output = cc_model(model_inputs[0])
        vcc[data_idx, sample_idx] = pem_output['cathode_potential']

    with Parallel(n_jobs=n_jobs, verbose=2) as ppool:
        for data_idx, data_tuple in enumerate(exp_data):
            # Run Monte Carlo
            print(f'Running Monte Carlo for PB={data_tuple[1]["background_pressure_Torr"]} Torr')
            ppool(delayed(run_batch)(data_idx, sample_idx, vcc_pred) for sample_idx in range(num_samples))

    # Compare to experimental data
    idx = np.argsort(data[:, 4])
    fig, ax = plt.subplots()
    yerr = 0.3
    ax.errorbar(data[idx, 4], data[idx, 5], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                label='Experimental', markersize=5)
    plot_qoi(ax, data[idx, 4], vcc_pred[idx, :], 'Pressure [torr]', 'Cathode coupling voltage [V]', legend=True)
    ax.set_ylim([30, 34])
    fig.tight_layout()
    plt.show()

    # Clean up
    del vcc_pred
    os.remove(fname)


def test_feedforward_mc(num_samples, n_jobs=-1):
    """Test full feedforward PEM with forward propagation of uncertainty"""
    # Load global component nominal conditions and uncertainties (exclude model parameter uncertainty)
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json', exclude=['parameters'])
    thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json', exclude=['parameters'])
    plume_nominal, plume_uncertainty = parse_input_file('plume_input.json', exclude=['parameters'])
    uncertainty_list = [cc_uncertainty, thruster_uncertainty, plume_uncertainty]

    # Create output directories
    dir_name = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
    base_path = Path('../results/forward') / dir_name
    os.mkdir(base_path)
    os.mkdir(base_path / 'thrust')
    os.mkdir(base_path / 'ion_velocity')
    os.mkdir(base_path / 'ion_current_density')

    # Batch function to run in parallel
    def run_batch(job_num, batch_sizes, operating_conditions, save_dir):
        # Import juliacall process once for each process
        from juliacall import Main as jl
        jl.seval('using HallThruster')

        # Set operating conditions
        cc_input = cc_nominal.copy()
        thruster_input = thruster_nominal.copy()
        plume_input = plume_nominal.copy()
        set_inputs(cc_input, operating_conditions)
        set_inputs(thruster_input, operating_conditions)
        set_inputs(plume_input, operating_conditions)
        nominal_list = [cc_input, thruster_input, plume_input]

        # Get current batch_size and start_idx
        batch_size = batch_sizes[job_num]
        start_idx = 0
        for i in range(job_num):
            start_idx += batch_sizes[i]

        # Run batch_size samples
        for idx in range(batch_size):
            sample_num = start_idx + idx

            # Sample inputs
            model_inputs = input_sampler(nominal_list, uncertainty_list)
            pem_output = {}

            # Run models
            try:
                pem_output = feedforward_pem(model_inputs, jl=jl)
            except ModelRunException as e:
                tb_str = traceback.format_exc()
                pem_output['Exception'] = tb_str
                data_write(pem_output, f'mc_{sample_num}_exc.json', dir=save_dir)
                logging.warning(f'Failed iteration i={sample_num}: {e}')
            else:
                data_write(pem_output, f'mc_{sample_num}.json', dir=save_dir)

            print(f'Finished sample number: {sample_num}')

    # Setup batch sizes
    num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
    batch_sizes = [math.floor(num_samples / num_batches)] * num_batches
    for i in range(num_samples % num_batches):
        # Evenly distribute remaining samples to rest of workers
        batch_sizes[i] += 1

    # Thrust predictions
    data = spt100_data(exclude=['vcc', 'ion_velocity', 'ion_current_density'])
    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        for idx, data_tuple in enumerate(data):
            # Make directory for each pressure
            pb = data_tuple[1]['background_pressure_Torr']
            pb_path = base_path / 'thrust' / f'{pb}_torr'
            os.mkdir(pb_path)

            # Run Monte Carlo
            print(f'Running Monte Carlo for thrust; PB={pb} Torr')
            ppool(delayed(run_batch)(job_num, batch_sizes, data_tuple[1], pb_path)
                  for job_num in range(num_batches))
            print(f'Finished Monte Carlo for thrust; PB={pb} Torr')

    # Ion velocity predictions
    data = spt100_data(exclude=['vcc', 'thrust', 'ion_current_density'])
    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        for idx, data_tuple in enumerate(data):
            # Make directory for each pressure
            pb = data_tuple[1]['background_pressure_Torr']
            pb_path = base_path / 'ion_velocity' / f'{pb}_torr'
            os.mkdir(pb_path)

            # Run Monte Carlo
            print(f'Running Monte Carlo for ion velocity; PB={pb} Torr')
            ppool(delayed(run_batch)(job_num, batch_sizes, data_tuple[1], pb_path)
                  for job_num in range(num_batches))
            print(f'Finished Monte Carlo for ion velocity; PB={pb} Torr')

    # Ion current density predictions
    data = spt100_data(exclude=['vcc', 'thrust', 'ion_velocity'])
    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        for idx, data_tuple in enumerate(data):
            # Make directory for each pressure
            pb = data_tuple[1]['background_pressure_Torr']
            pb_path = base_path / 'ion_current_density' / f'{pb}_torr'
            os.mkdir(pb_path)

            # Run Monte Carlo
            print(f'Running Monte Carlo for ion current density; PB={pb} Torr')
            ppool(delayed(run_batch)(job_num, batch_sizes, data_tuple[1], pb_path)
                  for job_num in range(num_batches))
            print(f'Finished Monte Carlo for ion current density; PB={pb} Torr')


if __name__ == '__main__':
    N = 200
    # test_cc_forward(N, n_jobs=-1)
    # test_feedforward_mc(N, n_jobs=-1)

    # Plot results (most recent run)
    dir = Path('../results/forward')
    files = os.listdir(dir)
    most_recent = '1900'
    for f in files:
        if f > most_recent:
            most_recent = f
    base_path = dir / most_recent
    plot_mc_experimental(base_path)
    # plot_mc(base_path)
