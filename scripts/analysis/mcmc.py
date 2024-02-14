"""Script for surrogate-enabled maximum likelihood estimation (MLE) and MCMC sampling."""
from pathlib import Path
import pickle
import time

import numpy as np
import uqtils as uq
import matplotlib.pyplot as plt
from scipy.optimize import direct, minimize, differential_evolution, OptimizeResult
import skopt
import h5py
from joblib import Parallel

from hallmd.data.loader import spt100_data
from hallmd.models.thruster import uion_reconstruct
from hallmd.models.plume import jion_reconstruct
from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

OPTIMIZER_ITER = 1
START_TIME = 0
PROJECT_ROOT = Path('../..')
TRAINING = False
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-01-03T02.35.53' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')
DATA = spt100_data()
COMP = 'System'
THETA_VARS = [v for v in SURR[COMP].x_vars if v.param_type == 'calibration']
QOI_MAP = {'Cathode': ['V_cc'], 'Thruster': ['T', 'uion'], 'Plume': ['jion'], 'System': ['V_cc', 'T', 'uion', 'jion']}
QOIS = QOI_MAP.get(COMP)

# Parse experimental data
QOI_USE, QOI_CNT = [], []
XE_ARRAY = np.zeros((0, 3))  # (PB, Va, mdot_a)
# sigma_y = np.zeros((0,))
# Ny = np.zeros((0,))
for qoi in QOIS:
    exp_data = DATA.get(qoi)[0]
    XE_ARRAY = np.concatenate((XE_ARRAY, exp_data.get('x')), axis=0)
    # sigma_y = np.hstack((sigma_y, np.sqrt(exp_data.get('var_y').flatten())))
    # Ny = np.hstack((Ny, np.prod(exp_data.get('y').shape)))
    qoi_add = [qoi] if exp_data.get('loc', None) is None else \
        [str(v) for v in SURR.coupling_vars if str(v).startswith(qoi)]
    QOI_USE.extend(qoi_add)
    QOI_CNT.append(len(qoi_add))
NOMINAL = {'r_m': 1, 'PB': XE_ARRAY[:, 0], 'Va': XE_ARRAY[:, 1], 'mdot_a': XE_ARRAY[:, 2]}
CONSTANTS = {'calibration', 'other', 'operating'}

with open(model_config_dir() / 'plume_svd.pkl', 'rb') as fd:
    PLUME_SVD = pickle.load(fd)
with open(model_config_dir() / 'thruster_svd.pkl', 'rb') as fd:
    THRUSTER_SVD = pickle.load(fd)


def spt100_log_likelihood(theta, M=100, ppool=None):
    """Compute the log likelihood for the SPT-100 dataset."""
    # Sample inputs and compute model
    theta = np.atleast_1d(theta)
    sample_shape = theta.shape[:-1] + (M, XE_ARRAY.shape[0])
    NOMINAL.update({str(v): theta[..., i, np.newaxis, np.newaxis] for i, v in enumerate(THETA_VARS)})
    xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=NOMINAL, constants=CONSTANTS).astype(theta.dtype)
    ys = SURR.predict(xs, qoi_ind=QOI_USE, training=TRAINING, ppool=ppool)

    # Initialize/ignore the constant part of the likelihood
    # const = -np.sum(Ny) * np.log(M) - (1/2) * np.sum(Ny) * np.log(2*np.pi) - np.sum(np.log(sigma_y))
    # const = -np.log(M) - (1/2) * np.sum(Ny) * np.log(2*np.pi) - np.sum(np.log(sigma_y))
    # log_likelihood = np.ones((*theta.shape[:-1], len(qois)), dtype=theta.dtype) * const
    log_likelihood = np.empty((*theta.shape[:-1], M, len(QOIS)), dtype=theta.dtype)

    xe_idx = 0
    qoi_idx = 0
    for k, qoi in enumerate(QOIS):
        exp_data = DATA.get(qoi)[0]
        xe, ye, var_y, loc = [exp_data.get(ele, None) for ele in ('x', 'y', 'var_y', 'loc')]
        y_curr = ys[..., xe_idx:xe_idx + xe.shape[0], qoi_idx:qoi_idx + QOI_CNT[k]]     # (..., M, Ne, ydim)
        std = np.sqrt(var_y)

        xe_idx += xe.shape[0]
        qoi_idx += QOI_CNT[k]

        # Reconstruct and interpolate spatial qois (case-by-case basis)
        match qoi:
            case 'jion':
                _, y_curr = jion_reconstruct(y_curr, alpha=loc[:, 1], svd_data=PLUME_SVD)
            case 'uion':
                _, y_curr = uion_reconstruct(y_curr, z=loc, svd_data=THRUSTER_SVD)

        ye = ye.reshape(y_curr.shape[-2:]).astype(theta.dtype)
        std = std.reshape(y_curr.shape[-2:]).astype(theta.dtype)

        # Evaluate the log likelihood for this qoi
        log_likelihood[..., k] = np.sum(-0.5 * ((ye - y_curr) / std) ** 2, axis=(-1, -2))

    # Reduce and combine
    log_likelihood = np.sum(log_likelihood, axis=-1)    # (..., M)
    max_log_like = np.max(log_likelihood, axis=-1, keepdims=True)
    log_likelihood = np.squeeze(max_log_like, axis=-1) + np.log(np.sum(np.exp(log_likelihood - max_log_like), axis=-1))

    return np.atleast_1d(log_likelihood)  # (...,)


def spt100_log_prior(theta):
    """Compute the log prior of the calibration parameters"""
    theta = np.atleast_1d(theta)
    with np.errstate(invalid='ignore', divide='ignore'):
        l = []
        for i, v in enumerate(THETA_VARS):
            bds = v.bounds()
            prior = np.log(v.pdf(theta[..., i]))
            bds_idx = np.logical_or(theta[..., i] < bds[0], theta[..., i] > bds[1])
            prior[bds_idx] = -np.inf
            l.append(prior)
        return np.atleast_1d(np.sum(l, axis=0)).astype(theta.dtype)  # (...,)


def spt100_log_posterior(theta, M=200, ppool=None):
    """Compute the un-normalized log posterior of PEM v0 calibration params given the SPT-100 dataset"""
    post = spt100_log_prior(theta)
    inf_idx = post > -np.inf
    if np.any(inf_idx):
        post[inf_idx] += spt100_log_likelihood(theta[inf_idx, :], M=M, ppool=ppool)
    return post


def pdf_slice(pdfs=None, M=200, n_jobs=-1):
    """Plot 1d slices of the likelihood/prior/posterior function(s)"""
    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        pdf_map = {'Prior': lambda theta: spt100_log_prior(theta),
                   'Likelihood': lambda theta: spt100_log_likelihood(theta, M=M, ppool=ppool),
                   'Posterior': lambda theta: spt100_log_posterior(theta, M=M, ppool=ppool)}
        if pdfs is None:
            pdfs = ['Prior', 'Likelihood', 'Posterior']
        pdfs = [pdfs] if isinstance(pdfs, str) else pdfs
        # bds = [v.bounds() for v in THETA_VARS]
        x0 = [v.nominal for v in THETA_VARS]
        bds = [(val - 0.01*np.abs(val), val + 0.01*np.abs(val)) for val in x0]
        funcs = [pdf_map.get(pdf) for pdf in pdfs]
        fig, ax = uq.plot_slice(funcs, bds, x0=x0, N=15, random_walk=True,
                                xlabels=[v.to_tex(units=True) for v in THETA_VARS], ylabels=['Log PDF'], fun_labels=pdfs,
                                x_idx=list(np.arange(0, 4)))
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


def run_mle(optimizer='nelder-mead', M=100):
    """Compute maximum likelihood estimate for the PEM."""
    global START_TIME
    # nominal = {'u_n': 102.5, 'l_t': 19.9, 'vAN1': -1.66, 'vAN2': 29.44, 'delta_z': 2.47e-3, 'z0*': -0.205, 'p_u': 44.5}
    # nominal = {'T_ec': 4.707, 'u_n': 100.4, 'l_t': 3.497, 'vAN1': -1.088, 'vAN2': 10.78, 'delta_z': 0.38, 'z0*': -2.172e-2, 'p_u': 29.19}  # f(x)=3965
    # nominal = {'T_ec': 4.1, 'u_n': 101, 'l_t': 20, 'vAN1': -1.97, 'vAN2': 31, 'delta_z': 4.35e-4, 'z0*': -0.2, 'p_u': 54}  # f(x)=5470
    # nominal = {'c0': 0.5938, 'c1': 0.4363, 'c2': -8.55, 'c3': 0.2812, 'c4': 20.13, 'c5': 16.64}  # f(x) = 526
    nominal = {}
    bds = [v.bounds() for v in THETA_VARS]
    x0 = [nominal.get(str(v), v.nominal) for v in THETA_VARS]
    obj_fun = lambda theta: float(-spt100_log_likelihood(theta.astype(np.float32), M=M))

    print_str = f'{"TIME (min)":>10} {"ITERATION":>10} {"f(X)":>12} ' + ' '.join([f"{str(v):>10}" for v in THETA_VARS])
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
            obj_fun = lambda theta: -spt100_log_likelihood(theta.T.astype(np.float32), M=M)
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


def run_laplace(M=100):
    mle_pt = np.array([v.nominal for v in SURR.exo_vars if v.param_type == 'calibration'])
    obj_fun = lambda theta: spt100_log_posterior(theta, M=M)
    hess = uq.approx_hess(obj_fun, mle_pt)

    # Try to compute hess_inv and save
    res_dict = {'mle': mle_pt, 'hess': hess}
    try:
        hess_inv = np.linalg.pinv(-hess)
        res_dict['hess_inv'] = hess_inv
        if not uq.is_positive_definite(hess_inv):
            res_dict['hess_inv_psd'] = uq.nearest_positive_definite(hess_inv)
    except Exception as e:
        print(f'Exception when computing hess_inv: {e}')
    finally:
        with open('laplace-result.pkl', 'wb') as fd:
            pickle.dump(res_dict, fd)

    print(f'Laplace approximation finished!')


def show_laplace():
    with open('laplace-result.pkl', 'rb') as fd:
        res_dict = pickle.load(fd)
    map = res_dict['mle']
    cov = res_dict['hess_inv_psd']
    samples = uq.normal_sample(map, cov, size=100)
    dim = 6
    labels = [v.to_tex(units=True) for v in SURR.x_vars if v.param_type == 'calibration']
    z = spt100_log_posterior(samples, M=10)
    uq.ndscatter(samples[:, :dim], subplot_size=2, labels=labels[:dim], z=z)
    plt.show()


def run_mcmc(file='dram-system.h5', clean=False, n_jobs=1, M=100):
    if clean:
        with h5py.File(file, 'a') as fd:
            group = fd.get('mcmc', None)
            if group is not None:
                del fd['mcmc']

    nwalk, niter = 1, 10000

    cov_pct = {'T_ec': 0.15, 'PT': 0.2, 'P*': 0.08, 'V_vac': 0.01, 'vAN1': 0.01, 'z0*': 0.15, 'p_u': 0.2, 'c0': 0.05,
               'c1': 0.02, 'c2': 0.15, 'c3': 0.04, 'c4': 0.01, 'c5': 0.02} if TRAINING else \
        {'T_ec': 0.1, 'PT': 0.2, 'P*': 0.08, 'V_vac': 0.01, 'vAN1': 0.01, 'z0*': 0.05, 'p_u': 0.2, 'c0': 0.03,
         'c1': 0.02, 'c2': 0.15, 'c3': 0.04, 'c4': 0.01, 'c5': 0.01, 'vAN2': 0.01, 'l_t': 0.05, 'delta_z': 0.005,
         'u_n': 0.005}
    # p0 = np.array([(v.bounds()[0] + v.bounds()[1])/2 for v in THETA_VARS]).astype(np.float32)
    p0 = np.array([v.nominal for v in THETA_VARS]).astype(np.float32)
    p0[np.isclose(p0, 0)] = 1
    cov0 = np.eye(p0.shape[0]) * np.array([(cov_pct.get(str(v), 0.08) * np.abs(p0[i]) / 2)**2 for i, v in enumerate(THETA_VARS)])
    cov0 *= 0.7
    # p0 = uq.normal_sample(p0, cov0, nwalk).astype(np.float32)

    with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
        fun = lambda theta: spt100_log_posterior(theta, M=M, ppool=ppool)
        uq.dram(fun, p0, niter, cov0=cov0.astype(np.float32), filename=file, adapt_after=500, adapt_interval=50,
                eps=1e-6, gamma=0.1)


def show_mcmc(file='dram-system.h5', burnin=0.1):
    with h5py.File(file, 'r') as fd:
        samples = fd['mcmc/chain']
        accepted = fd['mcmc/accepted']
        cov = np.mean(np.array(fd['mcmc/cov']), axis=0)
        niter, nwalk, ndim = samples.shape
        samples = samples[int(burnin*niter):, ...]
        colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'lime', 'fuchsia', 'royalblue', 'sienna']
        # labels = [f'x{i}' for i in range(ndim)]
        labels = [str(v) for v in THETA_VARS]

        lags, autos, iac, ess = uq.autocorrelation(samples, step=5)
        print(f'Average acceptance ratio: {np.mean(accepted)/niter:.4f}')
        print(f'Average IAC: {np.mean(iac):.4f}')
        print(f'Average ESS: {np.mean(ess):.4f}')

        nchains = min(4, nwalk)
        nshow = min(10, ndim)

        # Integrated auto-correlation plot
        fig, ax = plt.subplots()
        for i in range(nshow):
            ax.plot(lags, np.mean(autos[:, :, i], axis=1), c=colors[i], ls='-', label=labels[i])
        uq.ax_default(ax, 'Lag', 'Auto-correlation', legend=True)

        # Mixing plots
        fig, axs = plt.subplots(nshow, 1, sharex=True)
        for i in range(nshow):
            ax = axs[i]
            for k in range(nchains):
                ax.plot(samples[:, k, i], colors[k], alpha=0.4, ls='-')
            ax.set_ylabel(labels[i])
        axs[-1].set_xlabel('Iterations')
        fig.set_size_inches(7, nshow*2)
        fig.tight_layout()

        # Marginal corner plot
        fig, ax = uq.ndscatter(samples[..., :nshow].reshape((-1, nshow)), subplot_size=2, labels=labels[:nshow], plot='hist',
                               cov_overlay=cov[:nshow, :nshow])
        plt.show()


if __name__ == '__main__':
    M = 1
    optimizer = 'nelder-mead'
    file = f'dram-{COMP.lower()}-{"train" if TRAINING else "test"}.h5'

    # pdf_slice(pdfs=['Posterior'], M=M)
    # run_mle(optimizer, M)
    # run_laplace(M=M)
    # show_laplace()
    run_mcmc(M=M, file=file)
    show_mcmc(file=file)
