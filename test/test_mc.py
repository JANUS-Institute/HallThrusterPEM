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
import pickle

sys.path.append('..')

# Custom imports
from models.pem import feedforward_pem
from models.cc import cathode_coupling_model_feedforward as cc_model
from models.plume import plume_pem, jion_reconstruct
from models.thruster import thruster_pem, uion_reconstruct
from utils import parse_input_file, data_write, ModelRunException, ax_default
from utils import batch_normal_sample, is_positive_definite, nearest_positive_definite
from surrogates.system import SystemSurrogate
from data.loader import spt100_data


def plot_qoi(ax, x, qoi, xlabel, ylabel, legend=False):
    """ Plot QOI with 5%, 50%, 95% percentiles against x
    x: (Nx)
    qoi: (Nx, Ns)
    """
    p5 = np.percentile(qoi, 5, axis=1)
    med = np.percentile(qoi, 50, axis=1)
    p95 = np.percentile(qoi, 95, axis=1)
    ax.plot(x, med, '-k', label='Model')
    ax.fill_between(x, p5, p95, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    ax_default(ax, xlabel, ylabel, legend=legend)


def plot_mc_experimental(Ns=1000):
    """Plot thrust, ion velocity, and ion current density against experimental data with UQ bounds"""
    exp_data = spt100_data()
    model = SystemSurrogate.load_from_file(Path('../results/surrogates/build_2023-08-11T23.09.43')
                                           / 'sys' / 'sys_final_edit.pkl')
    # Thrust
    data = exp_data['T']
    pb = data['x'][:, 0]
    idx = np.argsort(pb)
    Nx = data['x'].shape[0]
    xs = np.empty((Nx, Ns, len(model.exo_vars)))
    for i in range(Nx):
        nominal = {'PB': np.log10(data['x'][i, 0]), 'Va': data['x'][i, 1], 'mdot_a': data['x'][i, 2]}
        xs[i, :, :] = model.sample_inputs((Ns,), use_pdf=True, nominal=nominal)
    ys = np.squeeze(model(xs, qois=['T']), axis=-1)
    fig, ax = plt.subplots()
    yerr = 2 * np.sqrt(data['noise_var'])
    ax.errorbar(pb, data['y']*1000, yerr=yerr*1000, fmt='or', capsize=3, markerfacecolor='none',
                label='Experimental', markersize=5)
    plot_qoi(ax, pb[idx], ys[idx, ...]*1000, 'Background pressure (Torr)', 'Thrust (mN)', legend=True)
    fig.tight_layout()
    plt.show()

    # Ion velocity
    # folders = os.listdir(base_path / 'ion_velocity')
    # Nx = len(folders)  # Number of pressure locations
    # Ns = len(os.listdir(base_path / 'ion_velocity' / folders[0]))  # Number of MC samples
    # folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))
    # pb = np.zeros(Nx)
    # n_grid = 152
    # z_m = np.zeros(n_grid)
    # pred = np.zeros((Nx, n_grid, Ns))
    #
    # # Load experimental data
    # data = spt100_data(exclude=['vcc', 'thrust', 'ion_current_density'])
    # data = sorted(data, key=lambda ele: ele[1]['background_pressure_Torr'])
    # ndata = len(data[0][1]['z_m'])
    # exp = np.zeros((Nx, ndata, 3))
    # for i, data_tuple in enumerate(data):
    #     exp[i, :, 0] = np.atleast_1d(data_tuple[1]['z_m'])
    #     exp[i, :, 1] = np.atleast_1d(data_tuple[2])
    #     exp[i, :, 2] = data_tuple[3]['value']
    #
    # # Load predictions
    # for i, folder in enumerate(folders):
    #     pb[i] = float(folder.split('_')[0])
    #     for j, filename in enumerate(os.listdir(base_path / 'ion_velocity' / folder)):
    #         if 'exc' not in str(filename):
    #             with open(base_path / 'ion_velocity' / folder / filename, 'r') as fd:
    #                 data = json.load(fd)
    #                 z_m[:] = np.atleast_1d(data['thruster']['output']['z'])
    #                 pred[i, :, j] = np.atleast_1d(data['thruster']['output']['ui_1'])
    #
    # # Plot ion velocity
    # fig, ax = plt.subplots(1, Nx, sharey='row')
    # for i in range(Nx):
    #     yerr = 2 * np.sqrt(exp[i, :, 2])
    #     ax[i].errorbar(exp[i, :, 0]*1000, exp[i, :, 1], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
    #                    label='Experimental', markersize=5)
    #     ylabel = 'Axial ion velocity [m/s]' if i == 0 else ''
    #     legend = i == 2
    #     plot_qoi(ax[i], z_m*1000, pred[i, :, :], 'Axial distance from anode [mm]', ylabel,
    #              legend=legend, linestyle='-k')
    #     ax[i].set_title(f'{pb[i]} torr')
    # fig.set_size_inches(9, 3)
    # fig.tight_layout()
    # plt.show()
    #
    # # Ion current density
    # folders = os.listdir(base_path / 'ion_current_density')
    # Nx = len(folders)  # Number of pressure locations
    # Ns = len(os.listdir(base_path / 'ion_current_density' / folders[0]))  # Number of MC samples
    # folders = sorted(folders, key=lambda ele: float(ele.split('_')[0]))
    # pb = np.zeros(Nx)
    #
    # # Load experimental data
    # data = spt100_data(exclude=['vcc', 'thrust', 'ion_velocity'])
    # data = sorted(data, key=lambda ele: ele[1]['background_pressure_Torr'])
    # ndata = len(data[0][1]['alpha_deg'])
    # exp = np.zeros((Nx, ndata, 3))
    # alpha_deg = np.zeros(ndata)
    # pred = np.zeros((Nx, ndata, Ns))
    # for i, data_tuple in enumerate(data):
    #     exp[i, :, 0] = np.atleast_1d(data_tuple[1]['alpha_deg'])
    #     exp[i, :, 1] = np.atleast_1d(data_tuple[2])
    #     exp[i, :, 2] = ((np.atleast_1d(data_tuple[2]) * data_tuple[3]['value']/100) / 2) ** 2
    #
    # # Load predictions
    # for i, folder in enumerate(folders):
    #     pb[i] = float(folder.split('_')[0])
    #     for j, filename in enumerate(os.listdir(base_path / 'ion_current_density' / folder)):
    #         if 'exc' not in str(filename):
    #             with open(base_path / 'ion_current_density' / folder / filename, 'r') as fd:
    #                 data = json.load(fd)
    #                 alpha_deg[:] = np.atleast_1d(data['plume']['input']['alpha_deg'])
    #                 pred[i, :, j] = np.atleast_1d(data['plume']['output']['ion_current_density'])
    #
    # # Plot ion velocity
    # fig, ax = plt.subplots(2, int(Nx/2), sharey='row', sharex='col')
    # idx = 0
    # for i in range(2):
    #     for j in range(int(Nx/2)):
    #         yerr = 2 * np.sqrt(exp[idx, :, 2])
    #         ax[i, j].errorbar(exp[idx, :, 0], exp[idx, :, 1]*0.1, yerr=yerr*0.1, fmt='or', capsize=3,
    #                           markerfacecolor='none', label='Experimental', markersize=5)
    #         xlabel = 'Angle from thruster centerline [deg]' if i == 1 else ''
    #         ylabel = 'Ion current density [$mA/cm^2$]' if j == 0 else ''
    #         legend = i == 0 and j == int(Nx/2) - 1
    #         plot_qoi(ax[i, j], alpha_deg, pred[idx, :, :]*0.1, xlabel, ylabel, legend=legend, linestyle='-k')
    #         ax[i, j].set_title(f'{pb[idx]} torr')
    #         ax[i, j].set_yscale('log')
    #         idx += 1
    # fig.set_size_inches(12, 6)
    # fig.tight_layout()
    # plt.show()


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
            elif uq_dict['uncertainty'] == 'loguniform_bds':
                lb, ub = uq_dict['value']
                input_dict[param] = 10 ** (np.random.rand() * (ub-lb) + lb)
            else:
                raise NotImplementedError

            # Update sample tracker
            sample_tracker[param] = input_dict.get(param)

        model_inputs.append(input_dict)

    return model_inputs


def laplace_sampler(nominal_list):
    # Load Laplace results
    base_path = Path('../results/laplace/laplace-2023-03-08T18.30.42.595069+00.00')
    with open(base_path / 'laplace-result.pkl', 'rb') as fd:
        data = pickle.load(fd)
    map = data['mle']
    cov = data['hess_inv']

    if not is_positive_definite(cov):
        cov = nearest_positive_definite(cov)  # zeroes out negative eigenvalues basically

    # Sample model parameters from Laplace approximation
    model_inputs = copy.deepcopy(nominal_list)
    samples = batch_normal_sample(map, cov)  # (1, dim)
    params = np.squeeze(samples, axis=0)  # (dim,)
    Te_c = params[0]    # Electron temperature at cathode (eV)
    V_vac = params[1]   # Vacuum coupling voltage at cathode (V)
    Pstar = params[2]   # Cathode coupling model parameter (torr)
    P_T = params[3]     # Cathode coupling model parameter (torr)
    u_n = params[4]     # Neutral velocity (m/s)
    c_w = params[5]     # Wall sheath loss coefficient
    c_AN1 = params[6]   # Anomalous transport coefficient
    c_AN2 = params[7]   # Anomalous transport coefficient (offset magnitude from cAN1)
    l_z = params[8]     # Inner-outer transition length (m)
    c0 = params[9]      # Plume model fit parameters
    c1 = params[10]
    c2 = params[11]
    c3 = params[12]
    c4 = params[13]
    c5 = params[14]
    set_inputs(model_inputs[0], {'cathode_electron_temp_eV': Te_c, 'V_vac': V_vac, 'Pstar': Pstar, 'P_T': P_T})
    set_inputs(model_inputs[1], {'cathode_electron_temp_eV': Te_c, 'neutral_velocity_m_s': u_n,
                                 'sheath_loss_coefficient': c_w, 'anom_coeff_1': c_AN1,
                                 'anom_coeff_2_mag_offset': c_AN2, 'inner_outer_transition_length_m': l_z})
    set_inputs(model_inputs[2], {'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5})

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

    # Load Laplace results
    base_path = Path('../results/laplace/laplace-2023-03-08T18.30.42.595069+00.00')
    with open(base_path / 'laplace-result.pkl', 'rb') as fd:
        laplace_data = pickle.load(fd)
    map = laplace_data['mle']
    cov = laplace_data['hess_inv']

    if not is_positive_definite(cov):
        cov = nearest_positive_definite(cov)  # zeroes out negative eigenvalues basically

    # Batch function to run in parallel
    def run_batch(data_idx, sample_idx, vcc):
        # Set operating conditions
        cc_input = cc_nominal.copy()
        operating_condition = exp_data[data_idx][1]
        set_inputs(cc_input, operating_condition)
        nominal_list = [cc_input]

        # Sample inputs and run model
        # model_inputs = input_sampler(nominal_list, uncertainty_list)

        # Sample model parameters from Laplace approximation
        model_inputs = copy.deepcopy(nominal_list)
        samples = batch_normal_sample(map, cov)  # (1, dim)
        params = np.squeeze(samples, axis=0)  # (dim,)
        Te_c = params[0]  # Electron temperature at cathode (eV)
        V_vac = params[1]  # Vacuum coupling voltage at cathode (V)
        Pstar = params[2]  # Cathode coupling model parameter (torr)
        P_T = params[3]  # Cathode coupling model parameter (torr)
        set_inputs(model_inputs[0], {'cathode_electron_temp_eV': Te_c, 'V_vac': V_vac, 'Pstar': Pstar, 'P_T': P_T})

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
    plot_qoi(ax, data[idx, 4], vcc_pred[idx, :], 'Background pressure [torr]', 'Cathode coupling voltage [V]', legend=True)
    ax.set_ylim([30, 34])
    fig.tight_layout()
    plt.show()

    # Clean up
    del vcc_pred
    os.remove(fname)


def test_feedforward_mc(num_samples, n_jobs=-1):
    """Test full feedforward PEM with forward propagation of uncertainty"""
    # Load global component nominal conditions and uncertainties (exclude model parameter uncertainty)
    exclude = ['parameters', 'operating', 'design', 'other']
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json', exclude=exclude)
    thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json', exclude=exclude)
    plume_nominal, plume_uncertainty = parse_input_file('plume_input.json', exclude=exclude)
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
            # model_inputs = input_sampler(nominal_list, uncertainty_list)
            model_inputs = laplace_sampler(nominal_list)
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


def test_plume_svd():
    """Test dimension reduction for plume ion current density"""
    def sampler(N):
        # Sample over entire input space
        c0 = np.random.rand(N, 1)
        c1 = np.random.rand(N, 1)*(0.9-0.1) + 0.1
        c2 = np.random.rand(N, 1)*(100 + 100) - 100
        c3 = np.random.rand(N, 1)*(1.570796) + 0
        c4 = 10 ** (np.random.rand(N, 1)*(22-18) + 18)
        c5 = 10 ** (np.random.rand(N, 1)*(18-14) + 14)
        PB = 10 ** (np.random.rand(N, 1)*(-3 + 8) - 8)
        IB0 = np.random.rand(N, 1)*(50) + 0
        sigma_cex = np.random.rand(N, 1)*(58e-20 - 51e-20) + 51e-20
        r = np.random.rand(N, 1)*(1.5 - 0.5) + 0.5
        x = np.concatenate((PB, c0, c1, c2, c3, c4, c5, sigma_cex, r, IB0), axis=-1)
        return x

    def save_svd():
        """Save svd results for a test set of N=1000"""
        x = sampler(6000)
        y = plume_pem(x, compress=False)['y']
        idx = ~np.isnan(y[:, 0]) & (np.nanmax(y, axis=-1) <= 1000)
        div_eff = y[idx, 0]
        div_angle = y[idx, 1] * (180 / np.pi)
        jion = y[idx, 2:]
        print(f'Number of good samples: {jion.shape[0]}')

        # Data matrix and SVD dimension reduction
        r_pct = 0.99
        N = 1000
        M = jion.shape[1]
        A = np.log10(jion[:N, :])
        u, s, vt = np.linalg.svd(A)  # (NxN), (NxM), (MxM)
        frac = np.cumsum(s / np.sum(s))
        idx = int(np.where(frac >= r_pct)[0][0])
        r = idx + 1  # Number of singular values to take
        print(f'r={r} with singular values {s[:r]} and cumsum {frac[:r]}')
        vtr = vt[:r, :]  # (r x M)
        save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
        with open(Path('../models/data') / 'plume_svd.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

        # Plot results for sanity check
        fig, ax = plt.subplots()
        ax.plot(s, '.k')
        ax.set_yscale('log')
        ax_default(ax, 'Index', 'Singular value', legend=False)
        plt.show()

        fig, ax = plt.subplots(1, 2)
        ax[0].hist(div_eff, color='r', edgecolor='black', linewidth=1.2, density=True)
        ax[1].hist(div_angle, color='r', edgecolor='black', linewidth=1.2, density=True)
        ax_default(ax[0], 'Divergence efficiency', '', legend=False)
        ax_default(ax[1], 'Divergence angle', '', legend=False)
        fig.set_size_inches(7, 3)
        fig.tight_layout()
        plt.show()
        fig, ax = plt.subplots()
        alpha = np.linspace(0, 90, 100)
        lb = np.percentile(jion, 5, axis=0)
        mid = np.percentile(jion, 50, axis=0)
        ub = np.percentile(jion, 95, axis=0)
        ax.plot(alpha, mid, '-k')
        for i in range(4):
            ax.plot(alpha, jion[i, :], linestyle='--')
        ax.fill_between(alpha, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
        ax.set_yscale('log')
        ax_default(ax, 'Angle from thruster centerline (deg)', 'Ion current density ($A/m^2$)', legend=False)
        plt.show()

    # Test reconstruction against experimental data
    def reconstruct():
        jion_data = np.loadtxt(Path('../data/spt100/jion_dataset4.csv'), delimiter=',', skiprows=1)
        Np = 8
        pb = jion_data[:, 3].reshape((Np, -1))
        r_m = jion_data[:, 4].reshape((Np, -1))
        alpha_deg = jion_data[:, 5].reshape((Np, -1))
        jion_exp = jion_data[:, 6].reshape((Np, -1))  # mA/cm^2
        jion_var = jion_data[:, 7].reshape((Np, -1))
        cutoff = np.where(alpha_deg[0, :] == 90)[0][0] + 1
        alpha_rad = alpha_deg[0, :cutoff] * (np.pi/180)

        # Set inputs
        x = np.zeros((Np, 10))
        x[:, 0] = pb[:, 0]      # Torr
        x[:, 1] = 0.5073
        x[:, 2] = 0.3031
        x[:, 3] = 9.73
        x[:, 4] = 0.261
        x[:, 5] = 3.17e19
        x[:, 6] = 1.09e16
        x[:, 7] = 55e-20
        x[:, 8] = r_m[:, 0]     # m
        x[:, 9] = 3.6           # A
        pem_res = plume_pem(x, compress=True)['y']   # (Np, 2+r)
        _, pem_interp = jion_reconstruct(pem_res[:, 2:], alpha=alpha_rad)  # (Np, M)
        pem_interp = pem_interp * (1000) * (1 / 100 ** 2)  # mA/cm^2

        fig, ax = plt.subplots(2, 4, sharey='all', sharex='all')
        fig.set_size_inches(12, 6)
        idx = 0
        for j in range(2):
            for i in range(4):
                # Plot model against experimental data
                yerr = 2 * np.sqrt(jion_var[idx, :cutoff])
                ax[j, i].errorbar(alpha_deg[idx, :cutoff], jion_exp[idx, :cutoff], yerr=yerr, fmt='or', capsize=3,
                                  markerfacecolor='none', label='Experimental', markersize=5)
                ax[j, i].plot(alpha_deg[idx, :cutoff], pem_interp[idx, :], '-k', label='PEM')
                ax[j, i].set_title(f'{pb[idx, 0]:.02} Torr')
                ax[j, i].set_yscale('log')
                ax_default(ax[j, i], 'Angle from thruster centerline (deg)', 'Ion current density ($mA/cm^2$)')
                idx += 1
        fig.tight_layout()
        plt.show()

    save_svd()
    reconstruct()


def test_thruster_svd():
    """Test dimension reduction for thruster avg ion velocity profile"""
    def sampler(N):
        # Sample over entire input space
        PB = 10 ** (np.random.rand(N, 1) * (-3 + 8) - 8)
        Va = np.random.rand(N, 1) * (400-200) + 200
        mdot_a = np.random.rand(N, 1) * (7e-6 - 3e-6) + 3e-6
        Tec = np.random.rand(N, 1) * (5 - 1) + 1
        un = np.random.rand(N, 1) * (500 - 100) + 100
        cw = np.random.rand(N, 1) * (0.3 - 0.1) + 0.1
        lt = np.random.rand(N, 1) * (0.02 - 0.001) + 0.001
        scale = np.log10(np.e)
        van_1 = 10 ** (np.random.rand(N, 1) * (-1 + 3) - 3)
        van_2 = np.random.rand(N, 1) * (100 - 2) + 2
        # van_1 = np.random.lognormal(mean=(1/scale)*(-3), sigma=np.sqrt((1/scale**2) * 0.25), size=(N, 1))
        # van_2 = np.random.randn(N, 1)*np.sqrt(0.25) + 1
        l_cathode = np.random.rand(N, 1) * (0.09 - 0.07) + 0.07
        Ti = np.random.rand(N, 1) * (1200 - 800) + 800
        Tn = np.random.rand(N, 1) * (320 - 280) + 280
        Tb = np.random.rand(N, 1) * (320 - 280) + 280
        Vcc = np.random.rand(N, 1) * (60 - 0) + 0
        x = np.concatenate((PB, Va, mdot_a, Tec, un, cw, lt, van_1, van_2, l_cathode, Ti, Tn, Tb, Vcc), axis=-1)
        return x

    def save_svd():
        """Save svd results for a test set of N=1000"""
        x = sampler(1500)
        alpha = (3, 1)
        timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
        save_dir = Path('../results/svd') / timestamp
        os.mkdir(save_dir)
        ret = thruster_pem(x, alpha, output_dir=save_dir, compress=False, n_jobs=-1)
        y = ret.get('y')
        idx = ~np.isnan(y[:, 0])
        IB0 = y[idx, 0]
        T = y[idx, 1]
        eta_v = y[idx, 2]
        eta_c = y[idx, 3]
        eta_m = y[idx, 4]
        u_avg = y[idx, 5]
        uion = y[idx, 6:]

        # Data matrix and SVD dimension reduction
        N = 1000
        r_pct = 0.99
        M = uion.shape[1]
        A = uion[:N, :]
        u, s, vt = np.linalg.svd(A)  # (NxN), (NxM), (MxM)
        frac = np.cumsum(s / np.sum(s))
        idx = int(np.where(frac >= r_pct)[0][0])
        r = idx + 1  # Number of singular values to take
        print(f'r={r} with singular values {s[:r]} and cumsum {frac[:r]}')
        vtr = vt[:r, :]  # (r x M)
        save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
        with open(Path('../models/data') / 'thruster_svd.pkl', 'wb') as fd:
            pickle.dump(save_dict, fd)

        # Plot results for sanity check
        fig, ax = plt.subplots()
        ax.plot(s, '.k')
        ax.set_yscale('log')
        ax_default(ax, 'Index', 'Singular value', legend=False)
        fig.savefig(str(Path('../results/figs')/'svd.png'), dpi=300, format='png')
        plt.show()
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(IB0, color='r', edgecolor='black', linewidth=1.2, density=True)
        ax[1].hist(T*1000, color='r', edgecolor='black', linewidth=1.2, density=True)
        ax_default(ax[0], 'Total beam current at exit (A)', '', legend=False)
        ax_default(ax[1], 'Thrust (mN)', '', legend=False)
        fig.set_size_inches(7, 3)
        fig.tight_layout()
        plt.show()
        fig.savefig(str(Path('../results/figs')/'qoi.png'), dpi=300, format='png')
        fig, ax = plt.subplots()
        z = np.linspace(0, 0.08, M)
        lb = np.percentile(uion, 5, axis=0)
        mid = np.percentile(uion, 50, axis=0)
        ub = np.percentile(uion, 95, axis=0)
        ax.plot(z*1000, mid, '-k')
        for i in range(4):
            ax.plot(z*1000, uion[i, :], linestyle='--')
        ax.fill_between(z*1000, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
        ax_default(ax, 'Distance from anode (mm)', 'Average ion velocity (m/s)', legend=False)
        plt.show()
        fig.savefig(str(Path('../results/figs')/'uion.png'), dpi=300, format='png')

    # Test reconstruction against experimental data
    def reconstruct():
        data = np.loadtxt(Path('../data/spt100/ui_dataset5.csv'), delimiter=',', skiprows=1)
        Np = 3
        data = data.reshape((Np, -1, 7))
        mdot_a = data[:, :, 2] * 1e-6    # kg/s
        pb = data[:, :, 3]
        z_m = data[:, :, 4]
        uion_exp = data[:, :, 5]
        uion_var = data[:, :, 6]

        # Set inputs
        x = np.zeros((Np, 14))
        x[:, 0] = pb[:, 0]  # Torr
        x[:, 1] = 300       # V
        x[:, 2] = mdot_a[:, 0]  # kg/s
        x[:, 3] = 2
        x[:, 4] = 64
        x[:, 5] = 0.2176
        x[:, 6] = 0.01093
        x[:, 7] = 0.0077361957
        x[:, 8] = 6.07
        x[:, 9] = 0.08
        x[:, 10] = 1000
        x[:, 11] = 300      # K
        x[:, 12] = 300      # K
        x[:, 13] = 30       # V
        ret = thruster_pem(x, (3, 1), output_dir='.', compress=True, n_jobs=3)  # (Np, 6+r)
        pem_res, files = ret['y'], ret['files']
        # with open('eval.pkl', 'rb') as fd:
        #     d = pickle.load(fd)
        #     pem_res = d['y']
        #     files = d['files']
        uion = pem_res[:, 6:]
        uion_interp = np.zeros(z_m.shape)   # (Np, M)

        fig, ax = plt.subplots()
        colors = ['r', 'b', 'g']
        for i in range(Np):
            _, interp = uion_reconstruct(uion[i, np.newaxis, :], z=z_m[i, :], L=0.08)
            uion_interp[i, :] = np.squeeze(interp)

            with open(files[i], 'r') as fd:
                pred = json.load(fd)
                z_pred = np.array(pred['output']['z'])
                uion_pred = np.array(pred['output']['ui_1'])

            # Plot model against experimental data
            yerr = 2 * np.sqrt(uion_var[i, :])
            ax.errorbar(z_m[i, :], uion_exp[i, :], yerr=yerr, fmt=f'o{colors[i]}', capsize=3,
                        markerfacecolor='none', label=f'{pb[i, 0]:.02} Torr', markersize=5)
            ax.plot(z_pred, uion_pred, f'-{colors[i]}', linewidth=0.5)
            ax.plot(z_m[i, :], uion_interp[i, :], f'--{colors[i]}')
        ax_default(ax, 'Distance from anode (m)', 'Avg ion velocity (m/s)')
        fig.tight_layout()
        plt.show()

    # save_svd()
    reconstruct()


if __name__ == '__main__':
    N = 200
    # test_cc_forward(N, n_jobs=-1)
    # test_feedforward_mc(N, n_jobs=-1)
    # test_plume_svd()
    # test_thruster_svd()

    # Plot results (most recent run)
    # dir = Path('../results/forward')
    # files = os.listdir(dir)
    # most_recent = '1900'
    # for f in files:
    #     if f > most_recent:
    #         most_recent = f
    # base_path = dir / most_recent
    plot_mc_experimental()
    # plot_mc(base_path)
