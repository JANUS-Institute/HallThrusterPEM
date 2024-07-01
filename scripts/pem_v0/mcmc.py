"""Script for surrogate-enabled maximum likelihood estimation (MLE) and MCMC sampling."""
from pathlib import Path
import pickle
import time

import numpy as np
import uqtils as uq
import matplotlib.pyplot as plt
import matplotlib
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
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-06-26T18.04.11' / 'single-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')
DATA = spt100_data()
COMP = 'System'
THETA_VARS = [v for v in SURR[COMP].x_vars if v.param_type == 'calibration']
QOI_MAP = {'Cathode': ['V_cc'], 'Thruster': ['uion'], 'Plume': ['T', 'jion'], 'System': ['V_cc', 'uion', 'T', 'jion']}
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
    qoi_add = ['Tc' if qoi == 'T' else qoi] if exp_data.get('loc', None) is None else \
        [str(v) for v in SURR.coupling_vars if str(v).startswith(qoi)]
    QOI_USE.extend(qoi_add)
    QOI_CNT.append(len(qoi_add))
NOMINAL = {'r_m': 1, 'PB': XE_ARRAY[:, 0], 'Va': XE_ARRAY[:, 1], 'mdot_a': XE_ARRAY[:, 2]}
CONSTANTS = {'calibration', 'other', 'operating'}
DISCHARGE_CURRENT = 4.5     # Experimental discharge current for all cases (A)
DISCHARGE_SIGMA = 0.2       # Be within this range of the correct value (smaller = more weight in likelihood)

with open(model_config_dir() / 'plume_svd.pkl', 'rb') as fd:
    PLUME_SVD = pickle.load(fd)
with open(model_config_dir() / 'thruster_svd.pkl', 'rb') as fd:
    THRUSTER_SVD = pickle.load(fd)


def spt100_log_likelihood(theta, M=100, ppool=None):
    """Compute the log likelihood for the SPT-100 dataset."""
    # Sample inputs and compute model
    theta = np.atleast_1d(theta)
    qoi_ind = QOI_USE if COMP == 'Cathode' else QOI_USE + ['I_D']
    sample_shape = theta.shape[:-1] + (M, XE_ARRAY.shape[0])
    NOMINAL.update({str(v): theta[..., i, np.newaxis, np.newaxis] for i, v in enumerate(THETA_VARS)})
    xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=NOMINAL, constants=CONSTANTS).astype(theta.dtype)
    ys = SURR.predict(xs, qoi_ind=qoi_ind, training=TRAINING, ppool=ppool)

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

    # Combine over QOIs
    log_likelihood = np.sum(log_likelihood, axis=-1)  # (..., M)
    if COMP != 'Cathode':
        # Add extra weight for discharge current
        log_likelihood += np.sum(-0.5 * ((DISCHARGE_CURRENT - ys[..., -1]) / DISCHARGE_SIGMA) ** 2, axis=-1)

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
                                x_idx=list(np.arange(0, 20)))
    # plt.show()
    plt.savefig('pdf_slice.png')


def mle_callback(xk, *args, **kwargs):
    """Callback function for Scipy optimizers"""
    global OPTIMIZER_ITER
    global START_TIME
    if isinstance(xk, OptimizeResult):
        x = xk.x
        res = xk.fun
    else:
        x = xk
        res = args[0] if len(args) > 0 else 0
        res = kwargs.get('convergence', res)

    x = np.atleast_1d(x)
    print_str = (f'{(time.time() - START_TIME)/60:>10.2f} {OPTIMIZER_ITER:>10} {float(res):>12.3E} ' +
                 ' '.join([f"{float(x[i]):>10.2E}" for i in range(x.shape[0])]))
    print(print_str)
    OPTIMIZER_ITER += 1


def run_mle(optimizer='nelder-mead', M=100):
    """Compute maximum likelihood estimate for the PEM."""
    global START_TIME
    # nominal = {'T_ec': 2.312, 'V_vac': 32.47, 'P*': 22.3, 'PT': 99.3, 'u_n': 311.0, 'c_w': 1.15, 'l_t': 5.90, 'f_n': 5.34, 
    #            'vAN1': -2.150, 'vAN2': 15.90, 'vAN3': 0.028, 'vAN4': 0.011, 'delta_z': 0.339, 'z0': -0.238, 'p0': 49.30, 
    #            'c0': 0.557, 'c1': 0.375, 'c2': -9.64, 'c3': 0.309, 'c4': 19.47, 'c5': 15.31}                                  # f(x)=1898, evolution, run 1

    # nominal = {'T_ec': 1.187, 'V_vac': 31.53, 'P*': 44.16, 'PT': 10.24, 'u_n': 162.4, 'c_w': 1.218, 'l_t': 4.802, 'f_n': .9857, 
    #            'vAN1': -2.054, 'vAN2': 12.65, 'vAN3': .0279, 'vAN4': .0101, 'delta_z': .3040, 'z0': -.2500, 'p0': 51.40, 
    #            'c0': .5653, 'c1': .3664, 'c2': -4.917, 'c3': .3013, 'c4': 19.20, 'c5': 14.22}                                 # f(x)=1875, nelder-mead anomalous variables, run 2

    # nominal = {'T_ec': 2.091, 'V_vac': 32.18, 'P*': 21.01, 'PT': 98.49, 'u_n': 278.4, 'c_w': .7844, 'l_t': 6.347, 'f_n': 4.289, 
    #            'vAN1': -2.500, 'vAN2': 31.76, 'vAN3': .0256, 'vAN4': .0101, 'delta_z': .2306, 'z0': -.0940, 'p0': 36.61, 
    #            'c0': .4998, 'c1': .3871, 'c2': -4.588, 'c3': .3041, 'c4': 19.48, 'c5': 15.02}                                 # f(x)=2545, nelder-mead thruster vars, run 3

    # nominal = {'T_ec': 1.011, 'V_vac': 31.76, 'P*': 35.54, 'PT': 10.15, 'u_n': 246.9, 'c_w': .8344, 'l_t': 4.331, 'f_n': 1.062, 
    #            'vAN1': -2.500, 'vAN2': 33.33, 'vAN3': .0262, 'vAN4': .0099, 'delta_z': .2015, 'z0': -.1158, 'p0': 33.94, 
    #            'c0': .4356, 'c1': .3830, 'c2': -6.532, 'c3': .3257, 'c4': 19.56, 'c5': 14.02}                                 # f(x)=2555, nelder-mead plume vars, run 4

    # nominal = {'T_ec': 1.32975724, 'V_vac': 31.4944035, 'P*': 37.6649643, 'PT': 10.0596097, 'u_n': 173.475606, 'c_w': 1.19056820, 
    #            'l_t': 6.15632359, 'f_n': .832893150, 'vAN1': -2.04501734, 'vAN2': 11.9583442, 'vAN3': .0283206049, 'vAN4': .0101167989, 
    #            'delta_z': .269274170, 'z0': -.2500, 'p0': 49.7019000, 'c0': .434697689, 'c1': .360025789, 'c2': -7.70634738, 
    #            'c3': .352032243, 'c4': 19.2993325, 'c5': 14.5840943}                                                          # f(x) = 1822, nelder-mead all vars, run 5

    nominal = {str(v): (v.bounds()[0] + v.bounds()[1])/2 for v in THETA_VARS}
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
    np.set_printoptions(threshold=np.inf)
    print(res.get('x'))
    np.set_printoptions(threshold=1000)


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

    nwalk, niter = 1, 50000

    cov_pct = {'T_ec': 0.04,
               'V_vac': 0.004,
               'P*': 0.06,
               'PT': 0.07,
               'u_n': 0.04,
               'c_w': 0.035,
               'l_t': 0.04,
               'f_n': 0.04,
               'vAN1': 0.006,
               'vAN2': 0.0085,
               'vAN3': 0.0065,
               'vAN4': 0.0065,
               'delta_z': 0.04,
               'z0': 0.06,
               'p0': 0.21,
               'c0': 0.05,
               'c1': 0.04,
               'c2': 0.09,
               'c3': 0.035,
               'c4': 0.005,
               'c5': 0.05}
    # cov_pct = {'T_ec': 0.04,      # Multi-Fidelity Settings
    #            'V_vac': 0.004,
    #            'P*': 0.06,
    #            'PT': 0.07,
    #            'u_n': 0.04,
    #            'c_w': 0.035,
    #            'l_t': 0.04,
    #            'f_n': 0.04,
    #            'vAN1': 0.006,
    #            'vAN2': 0.0085,
    #            'vAN3': 0.0065,
    #            'vAN4': 0.0065,
    #            'delta_z': 0.04,
    #            'z0': 0.06,
    #            'p0': 0.21,
    #            'c0': 0.05,
    #            'c1': 0.04,
    #            'c2': 0.09,
    #            'c3': 0.035,
    #            'c4': 0.005,
    #            'c5': 0.05}
    nominal = {'T_ec':    3.04387,
               'V_vac':   31.96449,
               'P_star':  31.44788,
               'PT':      10.54750,
               'u_n':     210.57903,
               'c_w':     1.15500,
               'l_t':     18.82385,
               'f_n':     1.23434,
               'vAN1':    -2.02342,
               'vAN2':    10.03596,
               'vAN3':    0.02687,
               'vAN4':    0.00907,
               'delta_z': 0.29058,
               'z0':      -0.22342,
               'p0':      47.53610,
               'c0':      0.40120,
               'c1':      0.35644,
               'c2':      -8.85125,
               'c3':      0.36195,
               'c4':      19.42760,
               'c5':      14.97537}
    # nominal = {'T_ec':    3.04387,        # MultiFidelity Nominals
    #            'V_vac':   30.96449,
    #            'P_star':  30.44788,
    #            'PT':      10.04750,
    #            'u_n':     100.57903,
    #            'c_w':     1.25500,
    #            'l_t':     18.82385,
    #            'f_n':     1.13434,
    #            'vAN1':    -2.01342,
    #            'vAN2':    10.03596,
    #            'vAN3':    0.02707,
    #            'vAN4':    0.00947,
    #            'delta_z': 0.29758,
    #            'z0':      -0.24342,
    #            'p0':      47.53610,
    #            'c0':      0.40120,
    #            'c1':      0.35644,
    #            'c2':      -8.85125,
    #            'c3':      0.36195,
    #            'c4':      19.42760,
    #            'c5':      14.97537}

    # p0 = np.array([(v.bounds()[0] + v.bounds()[1])/2 for v in THETA_VARS]).astype(np.float32)
    p0 = np.array([nominal.get(str(v), v.nominal) for v in THETA_VARS]).astype(np.float32)
    p0[np.isclose(p0, 0)] = 1
    cov0 = np.eye(p0.shape[0]) * np.array([(cov_pct.get(str(v), 0.08) * np.abs(p0[i]) / 2)**2 for i, v in enumerate(THETA_VARS)])
    cov0 *= 0.01
    # p0 = uq.normal_sample(p0, cov0, nwalk).astype(np.float32)

    # with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
    #     fun = lambda theta: spt100_log_posterior(theta, M=M, ppool=ppool)
    #     uq.dram(fun, p0, niter, cov0=cov0, filename=file, adapt_after=5000, adapt_interval=1500,
    #             eps=1e-6, gamma=0.15)

    fun = lambda theta: spt100_log_posterior(theta, M=M)
    uq.dram(fun, p0, niter, cov0=cov0, filename=file, adapt_after=5000, adapt_interval=1000, eps=1e-6, gamma=0.1)


def show_mcmc(file='dram-system.h5', burnin=0.1):
    with h5py.File(file, 'r') as fd:
        samples = fd['mcmc/chain']
        accepted = fd['mcmc/accepted']
        cov = np.mean(np.array(fd['mcmc/cov']), axis=0)
        niter, nwalk, ndim = samples.shape
        samples = samples[int(burnin*niter):, ...]
        colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'lime', 'fuchsia', 'royalblue', 'sienna', 
                  'salmon', 'yellow', 'crimson', 'navy', 'skyblue', 'peru', 'teal', 'indigo', 'slategray', 'gold', 'olive']
        # labels = [f'x{i}' for i in range(ndim)]
        labels = [str(v) for v in THETA_VARS]

        lags, autos, iac, ess = uq.autocorrelation(samples, step=20, maxlag=500)
        print(f'Average acceptance ratio: {np.mean(accepted)/niter:.4f}')
        print(f'Average IAC: {np.mean(iac):.4f}')
        print(f'Average ESS: {np.mean(ess):.4f}')

        nchains = min(4, nwalk)
        nshow = min(21, ndim)
        offset = 0
        j = offset*nshow
        nshow = min(nshow, ndim - j)

        # Integrated auto-correlation plot
        fig, ax = plt.subplots()
        for i in range(nshow):
            ax.plot(lags, np.mean(autos[:, :, j+i], axis=1), c=colors[i], ls='-', label=labels[j+i])
        uq.ax_default(ax, 'Lag', 'Auto-correlation', legend=True)
        plt.savefig('autocorr.png')

        # Mixing plots
        fig, axs = plt.subplots(nshow, 1, sharex=True)
        for i in range(nshow):
            ax = axs[i]
            for k in range(nchains):
                ax.plot(samples[:, k, j+i], colors[k], alpha=0.4, ls='-')
            ax.set_ylabel(labels[j+i])
        axs[-1].set_xlabel('Iterations')
        fig.set_size_inches(7, nshow*2)
        fig.tight_layout()
        plt.savefig('mixing_mcmc.png')

        # Marginal corner plot
        fig, ax = uq.ndscatter(samples[..., j:j+nshow].reshape((-1, nshow)), subplot_size=2, labels=labels[j:j+nshow],
                               plot2d='hist', cov_overlay=cov[j:j+nshow, j:j+nshow])
        # plt.show()
        plt.savefig('marginal_mcmc.png')


def journal_plots(file, burnin=0.1):
    """Make MCMC plots for journal."""
    with h5py.File(file, 'r') as fd:
        samples = fd['mcmc/chain']
        accepted = fd['mcmc/accepted']
        cov = np.mean(np.array(fd['mcmc/cov']), axis=0)
        niter, nwalk, ndim = samples.shape
        samples = samples[int(burnin*niter):, ...].reshape((-1, ndim))
        mincnt = int(0.0015 * samples.shape[0])
        bins = 21

        # Cathode marginals
        rc = {'axes.labelsize': 17, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.grid' : False}
        with plt.style.context("uqtils.default"):
            with matplotlib.rc_context(rc=rc):
                # Cathode marginals
                str_use = ['T_ec', 'V_vac', 'P*', 'PT']
                idx_use = sorted([THETA_VARS.index(v) for v in str_use])
                labels = [r'$T_e$ (eV)', r'$V_{vac}$ (V)', r'$P^*$ ($\mu$Torr)', r'$P_T$ ($\mu$Torr)']
                fig, ax = uq.ndscatter(samples[:, idx_use], subplot_size=2, labels=labels, plot1d='kde', plot2d='hex',
                                       cmap='viridis', cmin=mincnt, bins=bins)
                fig.savefig('mcmc-cathode.pdf', bbox_inches='tight', format='pdf')
                # plt.show()

                # Thruster marginals
                str_use = ['T_ec', 'u_n', 'c_w', 'l_t', 'f_n', 'vAN1', 'vAN2', 'vAN3', 'vAN4', 'delta_z', 'z0', 'p0']
                idx_use = sorted([THETA_VARS.index(v) for v in str_use])
                labels = [r'$T_e$ (eV)', r'$u_n$ (m/s)', r'$c_w$ (-)', r'$l_t$ (mm)', r'$f_n$ (m/s)', r'$a_1$ (-)', r'$a_2$ (-)', r'$a_3$ (-)', r'$a_4$ (-)', r'$\Delta z$ (-)', r'$z_0$ (-)', r'$p_0$ ($\mu$Torr)']
                fig, ax = uq.ndscatter(samples[:, idx_use], subplot_size=2, labels=labels, plot1d='kde', plot2d='hex',
                                       cmap='viridis', cmin=mincnt, bins=bins)
                fig.savefig('mcmc-thruster.pdf', bbox_inches='tight', format='pdf')
                # plt.show()

                # Plume marginals
                str_use = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
                idx_use = sorted([THETA_VARS.index(v) for v in str_use])
                labels = [r'$c_0$ (-)', r'$c_1$ (-)', r'$c_2$ (rad/Pa)', r'$c_3$ (rad)', r'$c_4$ $(10^x)$ ($m^{-3}$/Pa)',
                          r'$c_5$ $(10^x)$ ($m^{-3}$)']
                fmt = ['{x:.2f}' for i in range(len(str_use))]
                fig, ax = uq.ndscatter(samples[:, idx_use], subplot_size=2, labels=labels, plot1d='kde', plot2d='hex',
                                       cmap='viridis', tick_fmts=fmt, bins=bins, cmin=mincnt)
                fig.savefig('mcmc-plume.pdf', bbox_inches='tight', format='pdf')
                # plt.show()

                # Print 1d marginal stats
                print(f'Average acceptance ratio: {np.mean(accepted) / niter:.4f}')
                print(f'{"Variable": <10} {"Minimum": <20} {"5th percentile": <20} {"50th percentile": <20} '
                      f'{"95th percentile": <20} {"Maximum": <20} {"Std deviation": <20}')
                for i in range(ndim):
                    min = np.min(samples[:, i])
                    low = np.percentile(samples[:, i], 5)
                    med = np.percentile(samples[:, i], 50)
                    high = np.percentile(samples[:, i], 95)
                    max = np.max(samples[:, i])
                    std = np.std(samples[:, i])
                    print_str = ' '.join([f"{ele: <20.5f}" for ele in [min, low, med, high, max, std]])
                    print(f'{str(THETA_VARS[i]): <10} ' + print_str)


if __name__ == '__main__':
    M = 1
    optimizer = 'evolution'
    file = f'dram-{COMP.lower()}-{"train" if TRAINING else "test"}.h5'

    # pdf_slice(pdfs=['Posterior'], M=M)
    # run_mle(optimizer, M)
    # run_laplace(M=M)
    # show_laplace()
    # run_mcmc(M=M, file=file, clean=False)
    show_mcmc(file=file)
    # journal_plots(file)
