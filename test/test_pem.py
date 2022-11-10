# Standard imports
import matplotlib.pyplot as plt
import sys
import numpy as np
import logging
import scipy.optimize
import copy
from joblib import Parallel, delayed, cpu_count
import datetime
from datetime import timezone
import os
from pathlib import Path
import math
import time
import traceback

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.cc import cathode_coupling_model as cc_model
from models.thruster import hall_thruster_jl_model as thruster_model
from models.plume import current_density_model as plume_model
from utils import parse_input_file, data_write, ModelRunException, data_load


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


def run_nominal():
    # Load inputs
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
    thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
    plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')

    # Setup models
    model_inputs = [cc_nominal, thruster_nominal, plume_nominal]
    models = [cc_model, thruster_model, plume_model]

    # Run models
    model_out = {}
    for j, model in enumerate(models):
        model_out = model(model_inputs[j])

        if j < len(models) - 1:
            model_inputs[j + 1].update(model_out)

    # Quantities of interest
    r = np.atleast_1d(model_out['r'])
    alpha = np.atleast_1d(model_out['alpha'])
    j_ion = np.atleast_1d(model_out['ion_current_density'])

    # Plot results
    N = 50
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


def input_sampler(nominal_list, uncertainty_list):
    # Sample all inputs from uncertainty models (system input at i=-1)
    model_inputs = list()
    for i in range(len(nominal_list)):
        nominal_dict = nominal_list[i]
        uncertainty_dict = uncertainty_list[i]
        input_dict = copy.deepcopy(nominal_dict)

        for param, uq_dict in uncertainty_dict.items():
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

        model_inputs.append(input_dict)

    # Update all models with the system input samples
    sys_input = model_inputs[-1]
    for i, model_input in enumerate(model_inputs[:-1]):
        model_input.update(sys_input)

    # Update anomalous transport coefficients in thruster model
    anom_coeff_1_var = uncertainty_list[1]['anom_coeff_1']['value'][1]
    mag_offset = model_inputs[1]['anom_coeff_2'] * np.sqrt(anom_coeff_1_var)    # Offset order of magnitude
    model_inputs[1]['anom_coeff_2'] = model_inputs[1]['anom_coeff_1'] * max(1, 10 ** mag_offset)

    return model_inputs[:-1]


def test_feedforward_mc():
    # Load global component nominal conditions and uncertainties
    cc_nominal, cc_uncertainty = parse_input_file('cc_input.json')
    thruster_nominal, thruster_uncertainty = parse_input_file('thruster_input.json')
    plume_nominal, plume_uncertainty = parse_input_file('plume_input.json')

    # Load models
    models = [cc_model, thruster_model, plume_model]

    # Create output directory
    dir_name = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
    base_path = Path('../results') / dir_name
    os.mkdir(base_path)

    # Batch function to run in parallel
    def run_batch(job_num, batch_sizes, nominal_list, uncertainty_list, output_path):
        # Import juliacall process once for each process
        from juliacall import Main as jl
        jl.seval('using HallThruster')

        # Run batch_size samples
        even_bs = batch_sizes[0]  # First n_jobs-1 batch sizes are equal
        curr_batch_size = batch_sizes[job_num]
        for idx in range(curr_batch_size):
            # Allocate space for results
            results = dict()
            sample_num = even_bs * job_num + idx

            # Sample inputs
            model_inputs = input_sampler(nominal_list, uncertainty_list)
            for j, model_input in enumerate(model_inputs):
                results[f'model{j}'] = {'input': copy.deepcopy(model_input)}

            try:
                # Run models
                for j, model in enumerate(models):
                    model_out = model(model_inputs[j], jl=jl) if j == 1 else model(model_inputs[j])
                    results[f'model{j}']['output'] = copy.deepcopy(model_out)

                    if j < len(models)-1:
                        model_inputs[j+1].update(model_out)

            except ModelRunException as e:
                tb_str = traceback.format_exc()
                results['Exception'] = tb_str
                data_write(results, f'ff_mc_{sample_num}_exc.json', dir=output_path)
                logging.warning(f'Failed iteration i={idx}: {e}')
            else:
                data_write(results, f'ff_mc_{sample_num}.json', dir=output_path)

            print(f'Finished sample number: {sample_num}')

    # pressures = [0, 1.69e-6, 4.06e-6, 6.99e-6, 1.22e-5, 1.57e-5, 2.56e-5, 3.83e-5, 5.51e-5]
    pressures = [1.69e-6, 4.06e-6]
    n_jobs = -1
    N = 10

    for i, pb in enumerate(pressures):
        # Load background pressure into system inputs
        sys_data = data_load('system_input.json')
        sys_data['operating']['background_pressure_Torr']['nominal'] = pb
        data_write(sys_data, 'system_input.json')
        sys_nominal, sys_uncertainty = parse_input_file('system_input.json')
        nominal = [cc_nominal, thruster_nominal, plume_nominal, sys_nominal]
        uncertainty = [cc_uncertainty, thruster_uncertainty, plume_uncertainty, sys_uncertainty]

        # Make directory for each pressure
        pb_path = base_path / f'{pb}_torr'
        os.mkdir(pb_path)

        # Run Monte Carlo
        with Parallel(n_jobs=n_jobs, verbose=9) as parallel:
            print(f'Running Monte Carlo for PB={pb} Torr')
            num_batches = cpu_count() if n_jobs < 0 else min(n_jobs, cpu_count())
            batch_sizes = [math.floor(N/num_batches)] * num_batches
            batch_sizes[-1] += N % num_batches  # Add remaining samples in the last batch
            parallel(delayed(run_batch)(job_num, batch_sizes, nominal, uncertainty, pb_path)
                     for job_num in range(num_batches))

        time.sleep(1)
        print(f'Finished Monte Carlo for PB={pb} Torr')


def main():
    # Run nominal case
    # run_nominal()

    # Test cathode coupling model
    # test_cc()

    # Run Monte Carlo
    test_feedforward_mc()


if __name__ == '__main__':
    main()
