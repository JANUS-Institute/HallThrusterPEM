"""Script for surrogate-enabled maximum likelihood estimation (MLE)."""
from pathlib import Path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel
from scipy.optimize import direct, minimize, differential_evolution, OptimizeResult
from amisc.utils import approx_hess
import skopt

from hallmd.data.loader import spt100_data
from hallmd.models.thruster import uion_reconstruct
from hallmd.models.plume import jion_reconstruct
from hallmd.models.pem import pem_v0
from hallmd.utils import plot_slice

OPTIMIZER_ITER = 1
START_TIME = 0
PROJECT_ROOT = Path('../..')
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-01-03T02.35.53' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / 'sys_final.pkl')
DATA = spt100_data()


def spt100_log_likelihood(theta, data, surr, comp='System', M=100):
    """Compute the log likelihood for the SPT-100 dataset."""
    theta = np.atleast_1d(theta)
    qoi_map = {'Cathode': ['V_cc'], 'Thruster': ['T', 'uion'], 'Plume': ['jion'], 'System': ['V_cc', 'T', 'uion', 'jion']}
    theta_vars = [v for v in surr[comp].x_vars if v.param_type == 'calibration']
    qois = qoi_map.get(comp)

    # Parse experimental data
    qois_use, qoi_cnt = [], []
    xe_array = np.zeros((0, 3))  # (PB, Va, mdot_a)
    sigma_y = np.zeros((0,))
    Ny = np.zeros((0,))
    for qoi in qois:
        exp_data = data.get(qoi)[0]
        xe_array = np.concatenate((xe_array, exp_data.get('x')), axis=0)
        sigma_y = np.hstack((sigma_y, np.sqrt(exp_data.get('var_y').flatten())))
        Ny = np.hstack((Ny, np.prod(exp_data.get('y').shape)))
        qoi_add = [qoi] if exp_data.get('loc', None) is None else \
            [str(v) for v in surr.coupling_vars if str(v).startswith(qoi)]
        qois_use.extend(qoi_add)
        qoi_cnt.append(len(qoi_add))

    # Sample inputs and compute model
    Nx = xe_array.shape[0]
    sample_shape = theta.shape[:-1] + (M, Nx)
    nominal = {str(v): theta[..., i, np.newaxis, np.newaxis] for i, v in enumerate(theta_vars)}
    nominal.update({'r_m': 1, 'PB': xe_array[:, 0], 'Va': xe_array[:, 1], 'mdot_a': xe_array[:, 2]})
    constants = {'calibration', 'r_m'}
    xs = surr.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=constants).astype(theta.dtype)
    ys = surr.predict(xs, qoi_ind=qois_use, training=True)

    # Initialize the constant part of the likelihood
    const = -np.sum(Ny) * np.log(M) - (1/2) * np.sum(Ny) * np.log(2*np.pi) - np.sum(np.log(sigma_y))
    log_likelihood = np.ones((*theta.shape[:-1], len(qois)), dtype=theta.dtype) * const

    xe_idx = 0
    qoi_idx = 0
    for k, qoi in enumerate(qois):
        exp_data = data.get(qoi)[0]
        xe, ye, var_y, loc = [exp_data.get(ele, None) for ele in ('x', 'y', 'var_y', 'loc')]
        y_curr = ys[..., xe_idx:xe_idx + xe.shape[0], qoi_idx:qoi_idx + qoi_cnt[k]]     # (..., M, Ne, ydim)
        std = np.sqrt(var_y)

        xe_idx += xe.shape[0]
        qoi_idx += qoi_cnt[k]

        # Reconstruct and interpolate spatial qois (case-by-case basis)
        match qoi:
            case 'jion':
                _, y_curr = jion_reconstruct(y_curr, alpha=loc[:, 1])
            case 'uion':
                _, y_curr = uion_reconstruct(y_curr, z=loc)

        ye = ye.reshape(y_curr.shape[-2:]).astype(theta.dtype)
        std = std.reshape(y_curr.shape[-2:]).astype(theta.dtype)

        # Evaluate the log likelihood for this qoi
        log_like = -0.5 * ((ye - y_curr) / std) ** 2
        max_log_like = np.max(log_like, axis=-3, keepdims=True)
        log_like = np.squeeze(max_log_like, axis=-3) + np.log(np.sum(np.exp(log_like - max_log_like), axis=-3))
        log_likelihood[..., k] += np.sum(log_like, axis=(-1, -2))

    return np.sum(log_likelihood, axis=-1)  # (...,)


def likelihood_slice(comp='System', M=1000):
    """Plot 1d slices of the likelihood function"""
    theta_vars = [v for v in SURR[comp].x_vars if v.param_type == 'calibration']
    bds = [v.bounds() for v in theta_vars]
    x0 = [v.nominal for v in theta_vars]
    fun = lambda theta: spt100_log_likelihood(theta.astype(np.float32), DATA, SURR, comp=comp, M=M)
    fig, ax = plot_slice(fun, bds, x0=x0, N=20, random_walk=True, xlabels=[v.to_tex(units=True) for v in theta_vars],
                         ylabels=['Log likelihood'], x_idx=list(np.arange(0, len(bds))))
    plt.show()


def mle_callback(xk, **kwargs):
    """Callback function for Scipy optimizers"""
    global OPTIMIZER_ITER
    global START_TIME
    if isinstance(xk, OptimizeResult):
        x = xk.x
        res = xk.fun
    else:
        x = xk
        res = kwargs.get('convergence', 0)

    x = np.atleast_1d(x)
    print_str = (f'{(time.time() - START_TIME)/60:>10.2f} {OPTIMIZER_ITER:>10} {float(res):>12.3E} ' +
                 ' '.join([f"{float(x[i]):>10.2E}" for i in range(x.shape[0])]))
    print(print_str)
    OPTIMIZER_ITER += 1


def run_mle(optimizer='nelder-mead', comp='System', M=100):
    """Compute maximum likelihood estimate for the PEM."""
    global START_TIME
    theta_vars = [v for v in SURR[comp].x_vars if v.param_type == 'calibration']
    # nominal = {'u_n': 102.5, 'l_t': 19.9, 'vAN1': -1.66, 'vAN2': 29.44, 'delta_z': 2.47e-3, 'z0*': -0.205, 'p_u': 44.5}
    # nominal = {'T_ec': 4.707, 'u_n': 100.4, 'l_t': 3.497, 'vAN1': -1.088, 'vAN2': 10.78, 'delta_z': 0.38, 'z0*': -2.172e-2, 'p_u': 29.19}  # f(x)=3965
    # nominal = {'T_ec': 4.1, 'u_n': 101, 'l_t': 20, 'vAN1': -1.97, 'vAN2': 31, 'delta_z': 4.35e-4, 'z0*': -0.2, 'p_u': 54}  # f(x)=5470
    # nominal = {'c0': 0.5938, 'c1': 0.4363, 'c2': -8.55, 'c3': 0.2812, 'c4': 20.13, 'c5': 16.64}  # f(x) = 526
    nominal = {}
    bds = [v.bounds() for v in theta_vars]
    x0 = [nominal.get(str(v), v.nominal) for v in theta_vars]
    obj_fun = lambda theta: float(-spt100_log_likelihood(theta.astype(np.float32), DATA, SURR, comp=comp, M=M))

    print_str = f'{"TIME (min)":>10} {"ITERATION":>10} {"f(X)":>12} ' + ' '.join([f"{str(v):>10}" for v in theta_vars])
    print(print_str)

    # Scipy requires this function signature for `minimize`
    def minimize_callback(*, intermediate_result):
        return mle_callback(intermediate_result)

    res = None
    tol = 1e-4
    maxfev = 100
    START_TIME = time.time()
    match optimizer:
        case 'nelder-mead':
            res = minimize(obj_fun, np.array(x0), method='Nelder-Mead', bounds=bds, tol=tol,
                           options={'maxfev': maxfev, 'adaptive': True}, callback=minimize_callback)
        case 'evolution':
            popsize = 15
            obj_fun = lambda theta: -spt100_log_likelihood(theta.T.astype(np.float32), DATA, SURR, comp=comp, M=M)
            res = differential_evolution(obj_fun, bds, popsize=popsize, maxiter=maxfev, init='sobol',
                                         vectorized=True, polish=True, updating='deferred', callback=mle_callback)
        case 'direct':
            res = direct(obj_fun, bds, eps=1, maxiter=maxfev, callback=mle_callback, locally_biased=False)
        case 'powell':
            res = minimize(obj_fun, np.array(x0), method='Powell', bounds=bds,
                           options={'maxiter': maxfev, 'ftol': tol}, callback=mle_callback)
        # case 'bfgs':
        #     res = minimize(obj_fun, np.array(x0), method='L-BFGS-B', bounds=bds, tol=tol,
        #                    options={'maxfun': maxfev, 'ftol': tol, 'gtol': tol, 'finite_diff_rel_step': 0.01},
        #                    callback=minimize_callback)
        # case 'brute':
        #     Ns = 4
        #     x_grids = [np.linspace(b[0], b[1], Ns) for b in bds]
        #     grids = np.meshgrid(*x_grids)
        #     pts = np.vstack(list(map(np.ravel, grids))).T
        #     res = -spt100_log_likelihood(pts, DATA, SURR, comp=comp)
        #     i = np.argmin(res)
        #     res = {'fun': res[i], 'x': pts[i, :]}
        # case 'bopt':
        #     res = skopt.gp_minimize(obj_fun, bds, x0=x0, n_calls=100,
        #                             acq_func="gp_hedge", acq_optimizer='lbfgs', n_initial_points=50,
        #                             initial_point_generator='lhs', verbose=False, xi=0.01, noise=0.001)

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
    M = 10
    comp = 'Plume'
    optimizer = 'nelder-mead'

    # Plot slices of likelihood
    # likelihood_slice(comp, M)

    # Compute MLE
    run_mle(optimizer, comp, M)

    # Obtain Laplace approximation at the MLE point
    # run_laplace()
