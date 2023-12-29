"""Script for surrogate-enabled maximum likelihood estimation (MLE)."""
from pathlib import Path
import os
import datetime
from datetime import timezone
import pickle
import time

import numpy as np
from joblib import Parallel
from scipy.optimize import direct, minimize
from amisc.utils import approx_hess
import skopt

from hallmd.data.loader import spt100_data

OPTIMIZER_ITER = 0  # Global to track iteration of the optimizer


def spt100_log_likelihood(params, data, base_path='.', n_jobs=-1, ppool=None):
    """Compute the log likelihood for entire SPT-100 dataset."""
    # Allocate space
    global OPTIMIZER_ITER
    t1 = time.time()
    params = np.atleast_1d(params)
    Nd = len(data)
    fname = 'temp.dat'
    log_likelihood = np.memmap(fname, dtype='float64', shape=(Nd,), mode='w+')
    save_dir = Path(base_path) / f'iteration-{OPTIMIZER_ITER}'
    os.mkdir(save_dir)

    # TODO: Implement spt100 likelihood with surrogate

    OPTIMIZER_ITER += 1
    del log_likelihood
    os.remove(fname)
    res = None

    return res


def run_mle(optimizer='nelder-mead'):
    """Compute maximum likelihood estimate for the PEM."""
    # Create output directory
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().replace(':', '.')
    base_path = Path('../results/mle') / f'optimizer_run-{timestamp}'
    os.mkdir(base_path)
    data = spt100_data()

    bds, x0 = None, None  # TODO: implement mle optimization with surrogate

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
    run_mle(optimizer='nelder-mead')

    # Obtain Laplace approximation at the MLE point
    run_laplace()
