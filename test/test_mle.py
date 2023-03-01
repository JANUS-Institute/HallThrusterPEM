import numpy as np
from pathlib import Path
import sys
import os
import datetime
from datetime import timezone
from joblib import Parallel, delayed, cpu_count
import math
from scipy.optimize import direct, minimize
import pickle
import matplotlib.pyplot as plt
import skopt
import copy
import time
import pygtc
from matplotlib.ticker import FuncFormatter

sys.path.append('..')

from utils import parse_input_file, data_write, ax_default, data_load, approx_hess, batch_normal_sample
from utils import nearest_positive_definite, is_positive_definite, spt100_data
from models.pem import feedforward_pem

OPTIMIZER_ITER = 0  # Global to track iteration of the optimizer


def set_inputs(input_dict, inputs_to_set):
    for input_name, value in inputs_to_set.items():
        input_dict[input_name] = value


def spt100_log_likelihood(params, data, base_path='.', n_jobs=-1, ppool=None):
    """Compute the log likelihood for entire SPT-100 dataset"""
    # Allocate space
    global OPTIMIZER_ITER
    t1 = time.time()
    params = np.atleast_1d(params)
    Nd = len(data)
    fname = 'temp.dat'
    log_likelihood = np.memmap(fname, dtype='float64', shape=(Nd,), mode='w+')
    save_dir = Path(base_path) / f'iteration-{OPTIMIZER_ITER}'
    os.mkdir(save_dir)

    # Load model parameters from designated order
    Te_c = params[0]        # Electron temperature at cathode (eV)
    V_vac = params[1]       # Vacuum coupling voltage at cathode (V)
    Pstar = params[2]       # Cathode coupling model parameter (torr)
    P_T = params[3]         # Cathode coupling model parameter (torr)
    u_n = params[4]         # Neutral velocity (m/s)
    Ti = params[5]          # Ion temperature (K)
    c_w = params[6]         # Wall sheath loss coefficient
    c_AN1 = params[7]       # Anomalous transport coefficient
    c_AN2 = params[8]       # Anomalous transport coefficient (offset magnitude from cAN1)
    l_z = params[9]         # Inner-outer transition length (m)
    c0 = params[10]         # Plume model fit parameters
    c1 = params[11]
    c2 = params[12]
    c3 = params[13]
    c4 = params[14]
    c5 = params[15]

    # Load nominal inputs
    cc_nominal, _ = parse_input_file('cc_input.json')
    thruster_nominal, _ = parse_input_file('thruster_input.json')
    plume_nominal, _ = parse_input_file('plume_input.json')
    set_inputs(cc_nominal, {'cathode_electron_temp_eV': Te_c, 'V_vac': V_vac, 'Pstar': Pstar, 'P_T': P_T})
    set_inputs(thruster_nominal, {'cathode_electron_temp_eV': Te_c, 'neutral_velocity_m_s': u_n,
                                  'ion_temp_K': Ti, 'sheath_loss_coefficient': c_w, 'anom_coeff_1': c_AN1,
                                  'anom_coeff_2_mag_offset': c_AN2, 'inner_outer_transition_length_m': l_z})
    set_inputs(plume_nominal, {'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5})

    def run_batch(job_num, batch_sizes, data, log_likelihood):
        # Import juliacall process once for each job/cpu/batch
        from juliacall import Main as jl
        jl.seval('using HallThruster')

        # Get current batch_size and start_idx
        batch_size = batch_sizes[job_num]
        start_idx = 0
        for i in range(job_num):
            start_idx += batch_sizes[i]

        # Run batch_size samples
        for idx in range(batch_size):
            sample_num = start_idx + idx
            curr_data = data[sample_num]

            # Set operating conditions
            cc_input = cc_nominal.copy()
            thruster_input = thruster_nominal.copy()
            plume_input = plume_nominal.copy()
            set_inputs(cc_input, curr_data[1])
            set_inputs(thruster_input, curr_data[1])
            set_inputs(plume_input, curr_data[1])

            # Run model
            model_inputs = [cc_input, thruster_input, plume_input]
            pem_output = feedforward_pem(model_inputs, jl=jl)
            data_write(pem_output, f'pem_{sample_num}.json', dir=save_dir)

            # Switch value of experimental noise (constant or percent)
            y = np.atleast_1d(curr_data[2])
            noise = curr_data[3]
            var = 1
            if noise['noise_var'] == 'constant':
                var = noise['value']
            elif noise['noise_var'] == 'percent':
                var = ((noise['value'] / 100) * y / 2) ** 2
            else:
                raise NotImplementedError('This noise model is not implemented')

            # Switch computation of p(y|theta) on data type
            data_type = curr_data[0]
            if data_type == 'cc_voltage':
                vcc_pred = pem_output['cc']['output']['cathode_potential']
                log_likelihood[sample_num] = -np.log(np.sqrt(var*2*np.pi)) - 0.5*(y - vcc_pred)**2 / var
                # print(f'Pred: {vcc_pred:.2f} V, Actual: {y[0]:.2f} V, Log_like: {log_likelihood[sample_num]}')
            elif data_type == 'thrust':
                thrust_pred = pem_output['thruster']['output']['thrust'][0]
                log_likelihood[sample_num] = -np.log(np.sqrt(var*2*np.pi)) - 0.5*(y - thrust_pred)**2 / var
                # print(f'Pred: {thrust_pred*1000:.2f} mN, Actual: {y[0]*1000:.2f} mN, Log_like: {log_likelihood[sample_num]}')
            elif data_type == 'axial_ion_velocity':
                ui_pred = np.atleast_1d(pem_output['thruster']['output']['ui_1'])  # only singly charged ions for now
                z_pred = np.atleast_1d(pem_output['thruster']['output']['z'])
                z_grid = np.atleast_1d(curr_data[1]['z_m'])
                ui_interp = np.interp(z_grid, z_pred, ui_pred)
                log_like = -np.log(np.sqrt(var*2*np.pi)) - 0.5*(y - ui_interp)**2 / var
                log_likelihood[sample_num] = np.sum(log_like)
                # print(f'Pred: {ui_interp[-1]:.2f} m/s, Actual: {y[-1]:.2f} m/s, Log_like: {log_likelihood[sample_num]}')
            elif data_type == 'ion_current_density':
                jion_pred = np.atleast_1d(pem_output['plume']['output']['ion_current_density'])
                log_like = -np.log(np.sqrt(var * 2 * np.pi)) - 0.5 * (y - jion_pred) ** 2 / var
                log_likelihood[sample_num] = np.sum(log_like)

                # Assign arbitrarily low likelihood if 0 current is predicted everywhere (within numerical tolerance)
                if np.abs(np.sum(jion_pred)) < 1e-6:
                    log_likelihood[sample_num] = -1e20
            else:
                raise NotImplementedError('This data type has not been included yet')

    # Setup batch sizes
    num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
    batch_sizes = [math.floor(Nd / num_batches)] * num_batches
    for i in range(Nd % num_batches):
        # Evenly distribute remaining samples to rest of workers
        batch_sizes[i] += 1

    if ppool is None:
        with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
            ppool(delayed(run_batch)(job_num, batch_sizes, data, log_likelihood) for job_num in range(num_batches))
    else:
        ppool(delayed(run_batch)(job_num, batch_sizes, data, log_likelihood) for job_num in range(num_batches))

    # Print and return results
    res = float(np.sum(log_likelihood))
    with open(Path(base_path) / 'opt-progress.txt', 'a') as fd:
        print_str = f'{OPTIMIZER_ITER: 9d} {Te_c: 7.2f} {V_vac: 7.2f} {Pstar: 7.2E} {P_T: 7.2E} {u_n: 7.2f} ' \
                    f'{Ti: 7.2f} {c_w: 7.3f} {c_AN1: 7.2E} {c_AN2: 7.2f} {l_z: 7.3f} {c0: 7.4f} {c1: 7.4f} ' \
                    f'{c2: 7.2E} {c3: 7.2E} {c4: 7.2E} {c5: 7.2E} {res: 7.1f} Time: {time.time()-t1:.2f} s'
        print(print_str)
        fd.write(print_str + '\n')

    OPTIMIZER_ITER += 1
    del log_likelihood
    os.remove(fname)

    return res


def run_mle(optimizer='nelder-mead'):
    """Compute maximum likelihood estimate for the PEM"""
    # Create output directory
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
    base_path = Path('../results/mle') / f'optimizer_run-{timestamp}'
    os.mkdir(base_path)

    with open(base_path / 'opt-progress.txt', 'w') as fd:
        print_str = f'{"ITERATION":>9} {"Te_c":>7} {"V_vac":>7} {"Pstar":>7} {"P_T":>7} {"u_n":>7} {"Ti":>7} ' \
                    f'{"c_w":>7} {"c_AN1":>7} {"c_AN2":>7} {"l_z":>7} {"c0":>7} {"c1":>7} {"c2":>7} {"c3":>7} ' \
                    f'{"c4":>7} {"c5":>7} {"f(X)":>7}'
        print(print_str)
        fd.write(print_str + '\n')

    # All parameters
    x0 = np.array([1.99, 30.39, 2.87e-5, 3.89e-6, 64, 1000, 0.2, 0.0077, 0.78, 0.01, 0.5, 0.3, 9, 0.27, 3e19, 1e16])
    bds = [(1, 5), (0, 60), (-2e-5, 10e-5), (0, 15e-5), (100, 1000), (0, 1000), (0.1, 0.3), (0.000625, 0.0625),
           (0, 2), (0.001, 0.02), (0, 1), (0.1, 0.9), (-100, 100), (0, 1.5708), (1e18, 1e22), (1e14, 1e18)]

    # vcc+thruster params
    # x0 = np.array([3.06434933, 30.7864851, 2.93095367e-05, 9.22764083e-06, 100.460468, 500, 2.06540502e-01, 1.45925189e-03, 1.13832312, 1.05136163e-03])
    # bds = [(1, 5), (0, 60), (-2e-5, 10e-5), (0, 15e-5), (100, 1000), (0, 1000), (0.1, 0.3), (0.000625, 0.0625), (0, 2), (0.001, 0.02)]

    # Subset of thruster params (u_n, c_w, c_AN1, c_AN2, l_z)
    # x0 = np.array([100, 0.2, 0.00625, 1, 0.01])
    # bds = [(0, 500), (0.1, 0.3), (0.000625, 0.0625), (0, 2), (0.001, 0.02)]

    # Plume parameters (c0, ..., c5)
    # x0 = np.array([0.5, 0.3, -20, 0.27, 3e19, 1e16])
    # bds = [(0, 1), (0.1, 0.9), (-100, 100), (0, 1.5708), (1e18, 1e22), (1e14, 1e18)]

    # Load experimental data
    data = spt100_data()

    with Parallel(n_jobs=-1, verbose=0) as ppool:
        obj_fun = lambda theta: -spt100_log_likelihood(theta, data, base_path=str(base_path), ppool=ppool)

        res = None
        tol = 0.01
        maxfev = 1000
        if optimizer == 'nelder-mead':
            res = minimize(obj_fun, np.array(x0), method='Nelder-Mead', bounds=bds, tol=0.01,
                           options={'maxfev': maxfev, 'tol': tol, 'adaptive': True})
        elif optimizer == 'bopt':
            res = skopt.gp_minimize(obj_fun, bds, x0=x0, n_calls=100,
                                    acq_func="gp_hedge", acq_optimizer='lbfgs', n_initial_points=50,
                                    initial_point_generator='lhs', verbose=False, xi=0.01, noise=0.001)
        elif optimizer == 'direct':
            res = direct(obj_fun, bds, eps=tol, maxfun=maxfev)
        elif optimizer == 'powell':
            res = minimize(obj_fun, np.array(x0), method='Powell', bounds=bds, tol=tol,
                           options={'maxiter': maxfev, 'tol': tol})

    res_dict = {'x0': x0, 'bds': bds, 'res': res}
    with open(base_path / 'opt-result.pkl', 'wb') as fd:
        pickle.dump(res_dict, fd)
    print(f'Optimization finished!')
    print(res)


def pem_pred(params, base_path, n_jobs=-1):
    """Predict on full dataset with specified parameters and make figures"""
    params = np.atleast_1d(params)
    Te_c = params[0]    # Electron temperature at cathode (eV)
    V_vac = params[1]   # Vacuum coupling voltage at cathode (V)
    Pstar = params[2]   # Cathode coupling model parameter (torr)
    P_T = params[3]     # Cathode coupling model parameter (torr)
    u_n = params[4]     # Neutral velocity (m/s)
    Ti = params[5]      # Ion temperature (K)
    c_w = params[6]     # Wall sheath loss coefficient
    c_AN1 = params[7]   # Anomalous transport coefficient
    c_AN2 = params[8]   # Anomalous transport coefficient (offset magnitude from cAN1)
    l_z = params[9]     # Inner-outer transition length (m)
    c0 = params[10]     # Plume model fit parameters
    c1 = params[11]
    c2 = params[12]
    c3 = params[13]
    c4 = params[14]
    c5 = params[15]

    # Load nominal conditions and data
    cc_nominal, _ = parse_input_file('cc_input.json')
    thruster_nominal, _ = parse_input_file('thruster_input.json')
    plume_nominal, _ = parse_input_file('plume_input.json')
    set_inputs(cc_nominal, {'cathode_electron_temp_eV': Te_c, 'V_vac': V_vac, 'Pstar': Pstar, 'P_T': P_T})
    set_inputs(thruster_nominal, {'cathode_electron_temp_eV': Te_c, 'neutral_velocity_m_s': u_n,
                                  'ion_temp_K': Ti, 'sheath_loss_coefficient': c_w, 'anom_coeff_1': c_AN1,
                                  'anom_coeff_2_mag_offset': c_AN2, 'inner_outer_transition_length_m': l_z})
    set_inputs(plume_nominal, {'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5})
    data = spt100_data()

    # Look for or compute predictions
    save_dir = Path(base_path) / 'preds'
    if os.path.isdir(save_dir):
        # Use results that are already computed
        pass
    else:
        os.mkdir(save_dir)
        print('Computing predictions with optimal parameters...')

        def run_batch(job_num, batch_sizes, data):
            # Import juliacall process once for each job/cpu/batch
            from juliacall import Main as jl
            jl.seval('using HallThruster')

            # Get current batch_size and start_idx
            batch_size = batch_sizes[job_num]
            start_idx = 0
            for i in range(job_num):
                start_idx += batch_sizes[i]

            # Run batch_size samples
            for idx in range(batch_size):
                sample_num = start_idx + idx
                curr_data = data[sample_num]

                # Set operating conditions
                cc_input = cc_nominal.copy()
                thruster_input = thruster_nominal.copy()
                plume_input = plume_nominal.copy()
                set_inputs(cc_input, curr_data[1])
                set_inputs(thruster_input, curr_data[1])
                set_inputs(plume_input, curr_data[1])

                # Run model
                model_inputs = [cc_input, thruster_input, plume_input]
                pem_output = feedforward_pem(model_inputs, jl=jl)
                data_write(pem_output, f'pem_mle_{sample_num}.json', dir=save_dir)

        # Setup batch sizes
        Nd = len(data)
        num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
        batch_sizes = [math.floor(Nd / num_batches)] * num_batches
        for i in range(Nd % num_batches):
            # Evenly distribute remaining samples to rest of workers
            batch_sizes[i] += 1
        with Parallel(n_jobs=n_jobs, verbose=2) as ppool:
            ppool(delayed(run_batch)(job_num, batch_sizes, data) for job_num in range(num_batches))

    # Collect data and predictions into useful structs for plotting: ([pressures], [QoI], [var], [coord (optional)])
    cc = {'exp': ([], [], []), 'pred': ([], [])}
    thrust = {'exp': ([], [], []), 'pred': ([], [])}
    ui = {'exp': ([], [], [], []), 'pred': ([], [], [])}
    jion = {'exp': ([], [], [], []), 'pred': ([], [], [])}
    for i, data_tuple in enumerate(data):
        pem_output = data_load(f'pem_mle_{i}.json', dir=save_dir)

        # Switch on data type
        data_type = data_tuple[0]
        qoi = None
        if data_type == 'cc_voltage':
            qoi = cc
            qoi['pred'][0].append(pem_output['cc']['input']['background_pressure_Torr'])
            qoi['pred'][1].append(pem_output['cc']['output']['cathode_potential'])
        elif data_type == 'thrust':
            qoi = thrust
            qoi['pred'][0].append(pem_output['thruster']['input']['background_pressure_Torr'])
            qoi['pred'][1].append(pem_output['thruster']['output']['thrust'][0])
        elif data_type == 'axial_ion_velocity':
            qoi = ui
            qoi['exp'][-1].append(data_tuple[1]['z_m'])
            qoi['pred'][0].append(pem_output['thruster']['input']['background_pressure_Torr'])
            qoi['pred'][1].append(pem_output['thruster']['output']['ui_1'])
            qoi['pred'][-1].append(pem_output['thruster']['output']['z'])
        elif data_type == 'ion_current_density':
            qoi = jion
            qoi['exp'][-1].append(data_tuple[1]['alpha_deg'])
            qoi['pred'][0].append(pem_output['plume']['input']['background_pressure_Torr'])
            qoi['pred'][1].append(pem_output['plume']['output']['ion_current_density'])
            qoi['pred'][-1].append(pem_output['plume']['input']['alpha_deg'])
        else:
            raise NotImplementedError('This data type has not been included yet')

        # Switch on value of experimental noise (constant or percent)
        pb = data_tuple[1]['background_pressure_Torr']
        y = np.atleast_1d(data_tuple[2])
        noise = data_tuple[3]
        var = 1
        if noise['noise_var'] == 'constant':
            var = noise['value']
        elif noise['noise_var'] == 'percent':
            var = ((noise['value'] / 100) * y / 2) ** 2
        else:
            raise NotImplementedError('This noise model is not implemented')
        qoi['exp'][0].append(pb)
        qoi['exp'][1].append(data_tuple[2])
        var = np.atleast_1d(var)[0] if np.atleast_1d(var).shape[0] == 1 else list(np.atleast_1d(var))
        qoi['exp'][2].append(var)

    # Plot vcc data
    idx = np.argsort(cc['exp'][0])
    fig, ax = plt.subplots()
    yerr = 2 * np.sqrt(cc['exp'][2])
    ax.errorbar(cc['exp'][0], cc['exp'][1], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none', label='Experimental', markersize=5)
    ax.plot(np.atleast_1d(cc['pred'][0])[idx], np.atleast_1d(cc['pred'][1])[idx], '-k', label='PEM')
    ax.set_ylim([30, 34])
    ax_default(ax, 'Background pressure (Torr)', 'Cathode coupling voltage (V)')
    fig.tight_layout()
    plt.show()
    fig.savefig(str(Path(base_path) / 'vcc_pred.png'), dpi=300, format='png')

    # Plot ui data
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    idx = 0
    for i in range(2):
        for j in range(2):
            yerr = 2 * np.sqrt(ui['exp'][2][idx])
            ax[i, j].errorbar(np.atleast_1d(ui['exp'][-1][idx])*1000, ui['exp'][1][idx], yerr=yerr, fmt='or', capsize=3,
                              markerfacecolor='none', label='Experimental', markersize=5)
            ax[i, j].plot(np.atleast_1d(ui['pred'][-1][idx])*1000, ui['pred'][1][idx], '-k', label='PEM')
            ax[i, j].set_title(f'{ui["pred"][0][idx]:.02} Torr')
            ax_default(ax[i, j], 'Axial distance from anode (mm)', 'Average ion axial velocity ($m/s$)')
            idx += 1
    fig.tight_layout()
    plt.show()
    fig.savefig(str(Path(base_path) / 'ui_pred.png'), dpi=300, format='png')

    # Plot thrust data
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    xerr = np.atleast_1d(2 * np.sqrt(thrust['exp'][2]))
    thrust_exp = np.atleast_1d(thrust['exp'][1])
    thrust_pred = np.atleast_1d(thrust['pred'][1])
    x = np.linspace(np.min(thrust_exp), np.max(thrust_exp), 100)
    ax[0].errorbar(thrust_exp*1000, thrust_pred*1000, xerr=xerr*1000, fmt='ok', capsize=3, markerfacecolor='none', markersize=5)
    ax[0].plot(x*1000, x*1000, '--k')
    ax_default(ax[0], 'Actual thrust (mN)', 'Predicted thrust (mN)', legend=False)
    pct_error = ((thrust_pred - thrust_exp) / thrust_exp) * 100
    ax[1].hist(pct_error, density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
    ax_default(ax[1], r'Percent error in thrust (%)', '', legend=False)
    fig.tight_layout()
    plt.show()
    fig.savefig(str(Path(base_path) / 'thrust_pred.png'), dpi=300, format='png')

    # Plot ion current density data
    fig, ax = plt.subplots(2, 4, sharey='row')
    fig.set_size_inches(12, 6)
    idx = 0
    for i in range(2):
        for j in range(4):
            yerr = 2 * np.sqrt(jion['exp'][2][idx])
            ax[i, j].errorbar(np.atleast_1d(jion['exp'][-1][idx]), np.atleast_1d(jion['exp'][1][idx])*0.1,
                              yerr=yerr*0.1, fmt='or', capsize=3, markerfacecolor='none', label='Experimental',
                              markersize=5)
            ax[i, j].plot(np.atleast_1d(jion['pred'][-1][idx]), np.atleast_1d(jion['pred'][1][idx])*0.1, '-k',
                          label='PEM')
            ax[i, j].set_title(f'{jion["pred"][0][idx]:.02} Torr')
            ax_default(ax[i, j], 'Angle from thruster centerline (deg)', 'Ion current density ($mA/cm^2$)')
            idx += 1
    fig.tight_layout()
    fig.savefig(str(Path(base_path) / 'jion_pred.png'), dpi=300, format='png')
    plt.show()


def test_laplace():
    # Load MLE point
    base_path = Path('../results')
    with open(base_path / 'mle' / 'optimizer_run-2023-02-18T00.10.46.040190+00.00' / 'opt-result.pkl', 'rb') as fd:
        data = pickle.load(fd)
    opt_params = data['res'].x
    opt_params = np.atleast_1d([1.99, 30.39, 2.87e-5, 3.89e-6, opt_params[0], opt_params[1], opt_params[2],
                                opt_params[3], opt_params[4]])

    # Create output directory
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
    base_path = base_path / 'laplace' / f'laplace-{timestamp}'
    os.mkdir(base_path)

    with open(base_path / 'opt-progress.txt', 'w') as fd:
        # print_str = f'{"ITERATION":>9} {"u_n":>7} {"c_w":>7} {"c_AN1":>7} {"c_AN2":>7} {"l_z":>7} {"f(X)":>7}'
        print_str = f'{"ITERATION":>9} {"Te_c":>7} {"V_vac":>7} {"Pstar":>7} {"P_T":>7} {"u_n":>7} {"Ti":>7} {"c_w":>7} {"c_AN1":>7} {"c_AN2":>7} {"l_z":>7} {"f(X)":>7}'
        print(print_str)
        fd.write(print_str + '\n')

    # Load experimental data
    data = spt100_data()

    with Parallel(n_jobs=-1, verbose=0) as ppool:
        # Form log likelihood function
        obj_fun = lambda theta: -spt100_log_likelihood(theta, data, base_path=str(base_path), ppool=ppool)

        # Evaluate the hessian at the MLE point
        hess = approx_hess(obj_fun, opt_params)

    # Try to compute hess_inv and save
    res_dict = {'mle': opt_params, 'hess': hess}
    try:
        hess_inv = np.linalg.pinv(hess)
        res_dict['hess_inv'] = hess_inv
    except Exception as e:
        print(f'Exception when computing hess_inv: {e}')
    finally:
        with open(base_path / 'laplace-result.pkl', 'wb') as fd:
            pickle.dump(res_dict, fd)

    print(f'Laplace approximation finished!')


def show_laplace():
    # Load Laplace results
    base_path = Path('../results/laplace/laplace-2023-02-22T21.08.00.052683+00.00')
    with open(base_path / 'laplace-result.pkl', 'rb') as fd:
        data = pickle.load(fd)
    map = data['mle']
    cov = data['hess_inv']

    if not is_positive_definite(cov):
        cov = nearest_positive_definite(cov)

    N = 10000
    samples = batch_normal_sample(map, cov, N)  # (N, 1, dim)
    samples = np.squeeze(samples, axis=1)       # (N, dim)
    names = ['$T_{e,c}$ [eV]', '$V_{vac}$ [V]', '$P^*$ [Torr]', '$P_T$ [Torr]', '$u_n$ [m/s]', '$c_w$',
             '$c_{AN,1}$', '$c_{AN,2}$', '$l_z$ [m]']
    font = {'family': 'sans-serif', 'size': 7}
    fig = pygtc.plotGTC(chains=[samples],
                        figureSize=8,
                        paramNames=names,
                        tickShifts=(0, 0),
                        # truths=[map],
                        # truthLabels=['MLE'],
                        # truthColors=['red'],
                        # legendMarker='All',
                        colorsOrder=['blues', 'greens', 'oranges'],
                        nContourLevels=3,
                        nBins=30,
                        smoothingKernel=1,
                        customLabelFont=font,
                        customTickFont=font,
                        customLegendFont=font,
                        plotName='GTC.pdf'
                        )

    def tick_format_func(value, pos):
        if value > 1:
            return f'{value:.2f}'
        if value > 0.01:
            return f'{value:.4f}'
        if value < 0.01:
            return f'{value:.2E}'

    # Get indices of first column axes in triangle plot
    # (pygtc indexes top->bottom, left->right on 2d marginals first, then on 1d marginals top->bottom... super lame)
    ai = 0
    i = 1
    dim = cov.shape[0]      # dimension of the parameters
    dim_corner = dim - 1    # 2d-marginal corner plot has 1 less dimension than number of parameters
    first_col_idxs = []     # Build recursive sequence of the indices of the first column plots
    total_ele = dim_corner*(dim_corner+1)/2
    while ai < total_ele:
        first_col_idxs.append(ai)
        ai = first_col_idxs[-1] + i
        i += 1

    for i, ax in enumerate(fig.axes):
        ax.tick_params(axis='both', direction='in', labelsize=5, bottom=True, left=True, top=False, right=False,
                       length=2, width=1)

        # First column gets ytick labels
        if i in first_col_idxs:
            # ax.set_ylabel(names[first_col_idxs.index(i)], labelpad=0.1)
            ax.yaxis.set_major_formatter(FuncFormatter(tick_format_func))

        # Last row gets xtick labels
        if first_col_idxs[-1] <= i < first_col_idxs[-1] + dim_corner:
            # ax.set_xlabel(names[int(i-first_col_idxs[-1])], labelpad=.1)
            ax.xaxis.set_major_formatter(FuncFormatter(tick_format_func))

    # Very last 1d marginal on far right gets xtick labels too
    # fig.axes[-1].set_xlabel(names[-1], labelpad=.1)
    fig.axes[-1].xaxis.set_major_formatter(FuncFormatter(tick_format_func))
    fig.savefig(str(base_path / 'marginals.png'), dpi=300, format='png')

    plt.show()


if __name__ == '__main__':
    # Compute MLE
    run_mle(optimizer='nelder-mead')

    # Predict on most recent optimization run
    # dir = Path('../results/mle')
    # files = os.listdir(dir)
    # most_recent = 'optimizer_run-1900'
    # for f in files:
    #     if f > most_recent:
    #         most_recent = f
    # base_path = dir / most_recent
    # print(base_path)
    # with open(Path(base_path) / 'opt-result.pkl', 'rb') as fd:
    #     data = pickle.load(fd)
    # opt_params = data['res'].x
    # opt_params = [1.99, 30.39, 2.87e-5, 3.89e-6, opt_params[0], 1000, opt_params[1], opt_params[2], opt_params[3], opt_params[4]]
    # pem_pred(opt_params, base_path)

    # Predict on nominal run
    # base_path = Path('../results/nominal')
    # params = np.array([5, 31.1, 2.6e-5, 1.5e-5, 300.0, 1000.0, 0.15, 0.00625, 1, 0.01])  # nominal
    # # params = np.array([1.99, 30.39, 2.87e-5, 3.89e-6, 300.0, 1000.0, 0.15, 0.00625, 1, 0.01])  # vcc calibrated
    # pem_pred(params, base_path)

    # Predict on specific mle results
    # opt_folder = 'optimizer_run-2023-02-03T06.19.37.728933+00.00' # Calibration on vcc,ui datasets only
    # opt_folder = 'optimizer_run-2023-02-04T06.36.21.606711+00.00'   # Calibration on vcc,ui,thrust datasets
    # base_path = Path('../results/mle') / opt_folder
    # with open(base_path / 'opt-result.pkl', 'rb') as fd:
    #     data = pickle.load(fd)
    # opt_params = data['res'].x
    # pem_pred(opt_params, base_path)

    # Obtain Laplace approximation at the MLE point
    # test_laplace()
    # show_laplace()
