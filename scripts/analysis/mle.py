"""Script for surrogate-enabled maximum likelihood estimation (MLE)."""
from pathlib import Path
import pickle
import time

import numpy as np
from joblib import Parallel
from scipy.optimize import direct, minimize
from amisc.utils import approx_hess
import skopt

from hallmd.data.loader import spt100_data
from hallmd.models.thruster import uion_reconstruct
from hallmd.models.plume import jion_reconstruct
from hallmd.models.pem import pem_v0

PROJECT_ROOT = Path('../..')
OPTIMIZER_ITER = 1
START_TIME = 0


def spt100_log_likelihood(theta, data, surr, model='System', M=100):
    """Compute the log likelihood for the SPT-100 dataset."""
    qoi_map = {'Cathode': ['V_cc'], 'Thruster': ['T', 'uion'], 'Plume': ['jion']}
    global START_TIME
    global OPTIMIZER_ITER
    theta = np.atleast_1d(theta)

    if model == 'System':
        theta_vars = [v for v in surr.exo_vars if v.param_type == 'calibration']
        qois = ['V_cc', 'T', 'uion', 'jion']
    else:
        theta_vars = [v for v in surr[model].x_vars if v.param_type == 'calibration']
        qois = qoi_map.get(model)

    if OPTIMIZER_ITER == 1:
        print_str = f'{"TIME (min)":>10} {"ITERATION":>10} {"f(X)":>12} ' + ' '.join([f"{str(v):>10}" for v in theta_vars])
        print(print_str)

    nominal = {str(v): theta[..., i, np.newaxis] for i, v in enumerate(theta_vars)}
    constants = {'calibration', 'r_m'}
    sample_shape = theta.shape[:-1] + (M,)
    log_likelihood = np.empty((*theta.shape[:-1], len(qois)))

    for q, qoi in enumerate(qois):
        # Get experimental data
        exp_data = data.get(qoi)[0]
        xe, ye, var_y, loc = exp_data.get('x'), exp_data.get('y'), exp_data.get('var_y'), exp_data.get('loc', None)

        if loc is not None:
            qoi = [str(v) for v in surr.coupling_vars if str(v).startswith(qoi)]  # SVD coeffs for spatial quantities

        # Sample inputs and compute model
        Nx = xe.shape[0]
        xs = np.empty((*sample_shape, Nx, len(surr.exo_vars)))
        for i in range(Nx):
            nominal.update({'PB': xe[i, 0], 'Va': xe[i, 1], 'mdot_a': xe[i, 2]})
            if loc is not None and qoi[0].startswith('jion'):
                nominal.update({'r_m': loc[i, 0, 0]})
            xs[..., i, :] = surr.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=constants)
        ys = surr.predict(xs, qoi_ind=qoi, training=True)             # (..., M, Nx, ydim)
        std = np.sqrt(var_y)

        # Reconstruct and interpolate spatial qois (case-by-case basis)
        if loc is not None:
            std = np.squeeze(std, axis=-1)
            ye = np.squeeze(ye, axis=-1)
            if qoi[0].startswith('uion'):
                # Assumes z is same locations for all Nx
                _, ys = uion_reconstruct(ys, z=loc[0, :, 0])
            elif qoi[0].startswith('jion'):
                # Assumes alpha is same locations for all Nx
                _, ys = jion_reconstruct(ys, alpha=loc[0, :, 1])

        # Evaluate the log likelihood for this qoi
        const = np.squeeze(-np.sum(np.log(np.sqrt(2 * np.pi) * std), axis=(-1, -2))) - np.log(M)
        log_like = np.sum(-0.5 * ((ye - ys) / std) ** 2, axis=(-1, -2))  # (..., M)
        max_log_like = np.max(log_like, axis=-1, keepdims=True)
        log_likelihood[..., q] = (const + np.squeeze(max_log_like, axis=-1) +
                                  np.log(np.sum(np.exp(log_like - max_log_like), axis=-1)))

    ret = np.sum(log_likelihood, axis=-1)  # (...,)
    print_str = (f'{(time.time() - START_TIME)/60:>10.2f} {OPTIMIZER_ITER:>10} {float(np.mean(ret)):>12.3E} ' +
                 ' '.join([f"{float(np.mean(theta[..., i])):>10.2E}" for i in range(theta.shape[-1])]))
    print(print_str)
    OPTIMIZER_ITER += 1

    return ret


def run_mle(optimizer='nelder-mead', surr_dir='mf_2024-01-03T02.35.53', model='System'):
    """Compute maximum likelihood estimate for the PEM."""
    global START_TIME
    data = spt100_data()
    surr_dir = list((PROJECT_ROOT / 'results' / surr_dir / 'multi-fidelity').glob('amisc_*'))[0]
    surr = pem_v0(from_file=surr_dir / 'sys' / 'sys_final.pkl')
    theta_vars = [v for v in surr.exo_vars if v.param_type == 'calibration'] if model == 'System' else \
        [v for v in surr[model].x_vars if v.param_type == 'calibration']
    bds = [v.bounds() for v in theta_vars]
    x0 = [v.nominal for v in theta_vars]
    obj_fun = lambda theta: float(-spt100_log_likelihood(theta, data, surr, model=model))

    res = None
    tol = 0.01
    maxfev = 100
    START_TIME = time.time()
    if optimizer == 'nelder-mead':
        res = minimize(obj_fun, np.array(x0), method='Nelder-Mead', bounds=bds, tol=tol,
                       options={'maxfev': maxfev, 'adaptive': True})
    elif optimizer == 'bfgs':
        res = minimize(obj_fun, np.array(x0), method='L-BFGS-B', jac='2-point', bounds=bds, tol=tol)
    elif optimizer == 'brute':
        Ns = 5
        x_grids = [np.linspace(b[0], b[1], Ns) for b in bds]
        grids = np.meshgrid(*x_grids)
        pts = np.vstack(list(map(np.ravel, grids))).T
        res = -spt100_log_likelihood(pts, data, surr, model=model)
        i = np.argmin(res)
        res = {'fun': res[i], 'x': pts[i, :]}
    elif optimizer == 'bopt':
        res = skopt.gp_minimize(obj_fun, bds, x0=x0, n_calls=100,
                                acq_func="gp_hedge", acq_optimizer='lbfgs', n_initial_points=50,
                                initial_point_generator='lhs', verbose=False, xi=0.01, noise=0.001)
    elif optimizer == 'direct':
        res = direct(obj_fun, bds, eps=tol, maxfun=maxfev)
    elif optimizer == 'powell':
        res = minimize(obj_fun, np.array(x0), method='Powell', bounds=bds, tol=tol,
                       options={'maxiter': maxfev, 'tol': tol})

    # res_dict = {'x0': x0, 'bds': bds, 'res': res}
    # with open(Path(surr_dir) / 'mle-result.pkl', 'wb') as fd:
    #     pickle.dump(res_dict, fd)
    print(f'Optimization finished! Elapsed time: {time.time() - START_TIME:.2f} s')
    print(f'Opimization result: {res}')


def run_laplace():
    # Load experimental data
    base_path, opt_params = None, None  # TODO: Write laplace function using surrogate
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


if __name__ == '__main__':
    # Compute MLE
    run_mle(optimizer='brute', model='Thruster')

    # Obtain Laplace approximation at the MLE point
    # run_laplace()
