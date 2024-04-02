"""Script for surrogate-enabled Monte Carlo forward UQ analysis."""
from pathlib import Path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import h5py
import uqtils as uq
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.stats as st
from itertools import cycle
from amisc.rv import NormalRV

from hallmd.data.loader import spt100_data
from hallmd.models.plume import jion_reconstruct
from hallmd.models.thruster import uion_reconstruct
from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

PROJECT_ROOT = Path('../..')
TRAINING = False
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-03-07T01.53.07' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')
DATA = spt100_data()
COMP = 'System'
THETA_VARS = [v for v in SURR[COMP].x_vars if v.param_type == 'calibration']
QOI_MAP = {'Cathode': ['V_cc'], 'Thruster': ['T', 'uion'], 'Plume': ['jion'], 'System': ['V_cc', 'T', 'uion', 'jion']}
QOIS = QOI_MAP.get(COMP)
CONSTANTS = {"calibration", "delta_z", "r_m"}
DISCHARGE_CURRENT = 4.5  # A

with open(model_config_dir() / 'plume_svd.pkl', 'rb') as fd:
    PLUME_SVD = pickle.load(fd)
with open(model_config_dir() / 'thruster_svd.pkl', 'rb') as fd:
    THRUSTER_SVD = pickle.load(fd)

# Load MCMC results
with h5py.File(f'dram-{COMP.lower()}-{"train" if TRAINING else "test"}.h5', 'r') as fd:
    SAMPLES = fd['mcmc/chain']
    niter, nwalk, ndim = SAMPLES.shape
    SAMPLES = SAMPLES[int(0.1 * niter):, ...].reshape((-1, ndim))


def posterior_sampler(shape):
    if not isinstance(shape, tuple):
        shape = (shape,)
    i = np.random.randint(0, SAMPLES.shape[0], shape)
    return SAMPLES[i, :]


def prior_sampler(shape):
    samples = np.empty((*shape, len(THETA_VARS)))
    for i, v in enumerate(THETA_VARS):
        samples[..., i] = v.sample(shape)
    return samples


def run_models(Ns=1000):
    """Get full-model and surrogate predictions for prior/posterior samples and save to file"""
    t1 = time.time()
    with h5py.File(f'monte-carlo.h5', 'a') as fd:
        # Cathode coupling voltage
        print(f'Time: {(time.time()-t1)/60:.2f} min. Running cathode coupling voltage predictions...')
        data = DATA['V_cc'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        theta = posterior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING), axis=-1)
        fd.create_dataset('vcc/posterior/xmodel', data=xs)
        fd.create_dataset('vcc/posterior/ymodel', data=ys)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING), axis=-1)
        fd.create_dataset('vcc/prior/xmodel', data=xs)
        fd.create_dataset('vcc/prior/ymodel', data=ys)

        # Include discharge current (I_D) with all of these QoIs

        # Thrust (Diamant)
        print(f'Time: {(time.time() - t1) / 60:.2f} min. Running Diamant thrust predictions...')
        data = DATA['T'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        theta = posterior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        fd.create_dataset('thrust/posterior/xsurr', data=xs)
        fd.create_dataset('thrust/posterior/ysurr', data=ys)
        fd.create_dataset('thrust/posterior/xmodel', data=xmodel)
        fd.create_dataset('thrust/posterior/ymodel', data=ymodel)

        xs[..., 2] = data['x'][idx[0], 2]  # Alternate predictions with constant mdot_a
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel[..., 2] = data['x'][idx[0], 2]
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        fd.create_dataset('thrust/alternate/xsurr', data=xs)
        fd.create_dataset('thrust/alternate/ysurr', data=ys)
        fd.create_dataset('thrust/alternate/xmodel', data=xmodel)
        fd.create_dataset('thrust/alternate/ymodel', data=ymodel)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1])/2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        fd.create_dataset('thrust/prior/xsurr', data=xs)
        fd.create_dataset('thrust/prior/ysurr', data=ys)
        fd.create_dataset('thrust/prior/xmodel', data=xmodel)
        fd.create_dataset('thrust/prior/ymodel', data=ymodel)

        idx = np.argsort(pb)[-5]
        nominal = {'PB': pb[idx], 'Va': data['x'][idx, 1], 'mdot_a': data['x'][idx, 2]}
        xs = np.zeros((100, Ns, len(SURR.x_vars)))
        for i in range(100):
            theta = posterior_sampler(1)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
            xs[i, ...] = SURR.sample_inputs(Ns, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        fd.create_dataset('thrust-uq/posterior/xsurr', data=xs)
        fd.create_dataset('thrust-uq/posterior/ysurr', data=ys)

        # Thrust (Sankovic)
        print(f'Time: {(time.time() - t1) / 60:.2f} min. Running Sankovic thrust predictions...')
        data = DATA['T'][1]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': data['x'][:, 0], 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        theta = posterior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        fd.create_dataset('thrust-test/posterior/xsurr', data=xs)
        fd.create_dataset('thrust-test/posterior/ysurr', data=ys)
        fd.create_dataset('thrust-test/posterior/xmodel', data=xmodel)
        fd.create_dataset('thrust-test/posterior/ymodel', data=ymodel)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1]) / 2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        fd.create_dataset('thrust-test/prior/xsurr', data=xs)
        fd.create_dataset('thrust-test/prior/ysurr', data=ys)
        fd.create_dataset('thrust-test/prior/xmodel', data=xmodel)
        fd.create_dataset('thrust-test/prior/ymodel', data=ymodel)

        # Ion velocity
        print(f'Time: {(time.time() - t1) / 60:.2f} min. Running ion velocity predictions...')
        data = DATA['uion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        theta = posterior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        qois = [var for var in SURR.coupling_vars if str(var).startswith('uion')] + ['I_D']
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        vcc = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), use_model='best')
        thruster_in = np.concatenate((xmodel[..., SURR.graph.nodes['Thruster']['exo_in']], vcc), axis=-1)
        ymodel = SURR['Thruster']._model(thruster_in, alpha=(2, 2), compress=False, svd_data=THRUSTER_SVD)['y']
        fd.create_dataset('uion/posterior/xsurr', data=xs)
        fd.create_dataset('uion/posterior/ysurr', data=ys)
        fd.create_dataset('uion/posterior/xmodel', data=xmodel)
        fd.create_dataset('uion/posterior/ymodel', data=np.concatenate((ymodel[..., 7:], ymodel[..., 1:2]), axis=-1))

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1]) / 2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        vcc = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), use_model='best')
        thruster_in = np.concatenate((xmodel[..., SURR.graph.nodes['Thruster']['exo_in']], vcc), axis=-1)
        ymodel = SURR['Thruster']._model(thruster_in, alpha=(2, 2), compress=False, svd_data=THRUSTER_SVD)['y']
        fd.create_dataset('uion/prior/xsurr', data=xs)
        fd.create_dataset('uion/prior/ysurr', data=ys)
        fd.create_dataset('uion/prior/xmodel', data=xmodel)
        fd.create_dataset('uion/prior/ymodel', data=np.concatenate((ymodel[..., 7:], ymodel[..., 1:2]), axis=-1))

        # Ion current density
        print(f'Time: {(time.time() - t1) / 60:.2f} min. Running ion current density predictions...')
        data = DATA['jion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        theta = posterior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        qois = [var for var in SURR.coupling_vars if str(var).startswith('jion')] + ['I_D']
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        I_B0 = SURR.predict(xmodel, qoi_ind=['I_B0', 'I_D'], use_model='best')
        plume_in = np.concatenate((xmodel[..., SURR.graph.nodes['Plume']['exo_in']], I_B0[..., 0:1]), axis=-1)
        ymodel = SURR['Plume']._model(plume_in, compress=False, svd_data=PLUME_SVD)['y']
        alpha_g = np.linspace(0, np.pi / 2, 100)
        jion_g = ymodel[..., 1:]
        alpha_g2 = np.concatenate((-np.flip(alpha_g)[:-1], alpha_g))  # (2M-1,)
        jion_g2 = np.concatenate((np.flip(jion_g, axis=-1)[..., :-1], jion_g), axis=-1)  # (..., 2M-1)
        f = interp1d(alpha_g2, jion_g2, axis=-1)
        jion_interp = f(data['loc'][:, 1])  # (..., Nx)
        fd.create_dataset('jion/posterior/xsurr', data=xs)
        fd.create_dataset('jion/posterior/ysurr', data=ys)
        fd.create_dataset('jion/posterior/xmodel', data=xmodel)
        fd.create_dataset('jion/posterior/ymodel', data=np.concatenate((jion_interp, I_B0[..., 1:]), axis=-1))

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1]) / 2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        I_B0 = SURR.predict(xmodel, qoi_ind=['I_B0', 'I_D'], use_model='best')
        plume_in = np.concatenate((xmodel[..., SURR.graph.nodes['Plume']['exo_in']], I_B0[..., 0:1]), axis=-1)
        ymodel = SURR['Plume']._model(plume_in, compress=False, svd_data=PLUME_SVD)['y']
        alpha_g = np.linspace(0, np.pi / 2, 100)
        jion_g = ymodel[..., 1:]
        alpha_g2 = np.concatenate((-np.flip(alpha_g)[:-1], alpha_g))  # (2M-1,)
        jion_g2 = np.concatenate((np.flip(jion_g, axis=-1)[..., :-1], jion_g), axis=-1)  # (..., 2M-1)
        f = interp1d(alpha_g2, jion_g2, axis=-1)
        jion_interp = f(data['loc'][:, 1])  # (..., Nx)
        fd.create_dataset('jion/prior/xsurr', data=xs)
        fd.create_dataset('jion/prior/ysurr', data=ys)
        fd.create_dataset('jion/prior/xmodel', data=xmodel)
        fd.create_dataset('jion/prior/ymodel', data=np.concatenate((jion_interp, I_B0[..., 1:]), axis=-1))


def relative_l2(pred, targ):
    return np.sqrt(np.mean((pred - targ)**2) / np.mean(targ**2))


def spt100_monte_carlo(Ns=1000):
    """Plot `[V_cc, T, uion, jion]` against SPT-100 experimental data with UQ bounds."""
    file = Path('monte-carlo.h5')
    gray = (0.5, 0.5, 0.5)
    alpha = 0.2
    if not file.is_file():
        run_models(Ns)

    with h5py.File(file, 'r') as fd:
        # Cathode coupling voltage
        data = DATA['V_cc'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        ys_post = fd['vcc/posterior/ymodel']
        ys_prior = fd['vcc/prior/ymodel']
        yerr = 2 * np.sqrt(data['var_y'])
        ys_post_pred = np.squeeze(uq.normal_sample(np.array(ys_post)[..., np.newaxis], (np.mean(yerr)/2)**2), axis=-1)
        fig, ax = plt.subplots(1, 2)
        p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        prior_h = ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
        prior_h2, = ax[0].plot(10 ** pb[idx], med[idx], ls='--', c='k')
        exp_h = ax[0].errorbar(10 ** pb, data['y'], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        uq.ax_default(ax[0], 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=False)
        ax[0].set_xscale('log')
        leg = ax[0].legend([(prior_h, prior_h2), exp_h], ['Prior predictive', 'Experiment'],
                           loc='upper left', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        p5_pred, p95_pred = list(map(lambda x: np.percentile(ys_post_pred, x, axis=0), (5, 95)))
        p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        h1 = ax[1].fill_between(10 ** pb[idx], p95[idx], p95_pred[idx], alpha=alpha, edgecolor='b', facecolor='b')
        h2 = ax[1].fill_between(10 ** pb[idx], p5_pred[idx], p5[idx], alpha=alpha, edgecolor='b', facecolor='b')
        h3 = ax[1].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
        h4, = ax[1].plot(10 ** pb[idx], med[idx], ls='--', c='r')
        h5 = ax[1].errorbar(10 ** pb, data['y'], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        uq.ax_default(ax[1], 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=False)
        ax[1].set_xscale('log')
        leg = ax[1].legend([(h3, h4), h1, h5], ['Posterior predictive', 'Posterior w/noise', 'Experiment'],
                           loc='upper left', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-vcc.png', dpi=300, format='png')
        plt.show()

        # Thrust (Diamant)
        data = DATA['T'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        yerr = 2 * np.sqrt(data['var_y'])
        cov = np.expand_dims(data['var_y'], axis=(-1, -2))
        ys_post = fd['thrust/posterior/ysurr'][..., 0]
        ys_post_pred = np.squeeze(uq.normal_sample(np.array(ys_post)[..., np.newaxis], cov), axis=-1)
        ys_prior = fd['thrust/prior/ysurr'][..., 0]
        ym_post = fd['thrust/posterior/ymodel'][..., 0]
        ym_prior = fd['thrust/prior/ymodel'][..., 0]
        fig, ax = plt.subplots(1, 2, sharey='row')
        p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        h1 = ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
        h2, = ax[0].plot(10 ** pb[idx], med[idx], ls='--', c='k')
        h3, = ax[0].plot(10 ** pb[idx], ym_prior[idx]*1000, ls='-', c='k')
        h4 = ax[0].errorbar(10 ** pb, data['y'] * 1000, yerr=yerr*1000, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        ax[0].set_title('Prior predictive')
        uq.ax_default(ax[0], 'Background pressure (Torr)', 'Thrust (mN)', legend=False)
        ax[0].set_xscale('log')
        ax[0].set_ylim(top=125, bottom=68)
        leg = ax[0].legend([(h1, h2), h3, h4], ['Surrogate', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        p5_pred, p95_pred = list(map(lambda x: 1000 * np.percentile(ys_post_pred, x, axis=0), (5, 95)))
        p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        h1 = ax[1].fill_between(10 ** pb[idx], p95[idx], p95_pred[idx], alpha=alpha, edgecolor='b', facecolor='b')
        h2 = ax[1].fill_between(10 ** pb[idx], p5_pred[idx], p5[idx], alpha=alpha, edgecolor='b', facecolor='b')
        h3 = ax[1].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
        h4, = ax[1].plot(10 ** pb[idx], med[idx], ls='--', c='r')
        h5, = ax[1].plot(10 ** pb[idx], ym_post[idx]*1000, ls='-', c='r')
        h6 = ax[1].errorbar(10 ** pb, data['y'] * 1000, yerr=yerr*1000, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        ax[1].set_title('Posterior predictive')
        uq.ax_default(ax[1], 'Background pressure (Torr)', '', legend=False)
        leg = ax[1].legend([(h3, h4), h1, h5, h6], ['Surrogate', 'Surrogate w/noise', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        ax[1].set_xscale('log')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-thrust.png', dpi=300, format='png')
        plt.show()

        ys = fd['thrust-uq/posterior/ysurr'][..., 0] * 1000  # (100, 1000), CDF plots at PB=3.81E-5 Torr
        ys = np.sort(ys, axis=-1)
        cdfs = np.arange(1, ys.shape[-1] + 1) / ys.shape[-1]
        cycol = cycle('bgrcmk')
        fig, ax = plt.subplots()
        for i in range(0, ys.shape[0], 2):
            ax.plot(ys[i, :], cdfs, ls='-', c=gray, alpha=0.2)
        xy = (79.35, 0.2)
        ax.annotate('Epistemic uncertainty', xy, xytext=(xy[0]+0.63, xy[1]),
                    arrowprops={'arrowstyle': '<|-|>', 'linewidth': 1.4, 'color': 'k'})
        uq.ax_default(ax, 'Thrust (mN)', 'Aleatoric CDF', legend=False)
        fig.set_size_inches(5, 4)
        fig.tight_layout()
        fig.savefig('mc-thrust-cdf.png', dpi=300, format='png')
        plt.show()

        # Discharge current (Diamant)
        fig, ax = plt.subplots(1, 2, sharey='row')
        ys_post = fd['thrust/posterior/ysurr'][..., 1]
        ys_prior = fd['thrust/prior/ysurr'][..., 1]
        ym_post = fd['thrust/posterior/ymodel'][..., 1]
        ym_prior = fd['thrust/prior/ymodel'][..., 1]
        p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        h1 = ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
        h2, = ax[0].plot(10 ** pb[idx], med[idx], ls='--', c='k')
        h3, = ax[0].plot(10 ** pb[idx], ym_prior[idx], ls='-', c='k')
        h4 = ax[0].errorbar(10 ** pb, np.ones(pb.shape[0])*DISCHARGE_CURRENT, yerr=0.1*DISCHARGE_CURRENT,
                            fmt='ok', markersize=4, markerfacecolor='none', capsize=3)
        ax[0].set_title('Prior predictive')
        uq.ax_default(ax[0], 'Background pressure (Torr)', 'Discharge current (A)', legend=False)
        ax[0].set_xscale('log')
        ax[0].set_ylim([0, 35])
        leg = ax[0].legend([(h1, h2), h3, h4], ['Surrogate', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        h1 = ax[1].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
        h2, = ax[1].plot(10 ** pb[idx], med[idx], ls='--', c='r')
        h3, = ax[1].plot(10 ** pb[idx], ym_post[idx], ls='-', c='r')
        h4 = ax[1].errorbar(10 ** pb, np.ones(pb.shape[0])*DISCHARGE_CURRENT, yerr=0.1*DISCHARGE_CURRENT,
                            fmt='ok', markersize=4, markerfacecolor='none', capsize=3)
        ax[1].set_title('Posterior predictive')
        uq.ax_default(ax[1], 'Background pressure (Torr)', '', legend=False)
        leg = ax[1].legend([(h1, h2), h3, h4], ['Surrogate', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        ax[1].set_xscale('log')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-thrust-discharge.png', dpi=300, format='png')
        plt.show()

        # Thrust (Sankovic 1993)
        fig, ax = plt.subplots(1, 2, sharey='row')
        data = DATA['T'][1]
        ys_post = fd['thrust-test/posterior/ysurr'][..., 0]
        ys_prior = fd['thrust-test/prior/ysurr'][..., 0]
        ym_post = fd['thrust-test/posterior/ymodel'][..., 0]
        ym_prior = fd['thrust-test/prior/ymodel'][..., 0]
        xerr = 2 * np.sqrt(data['var_y']) * 1000
        p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        yerr = np.vstack((med - p5, p95 - med))
        ax[0].errorbar(data['y'] * 1000, med, yerr=yerr, xerr=xerr, fmt='or', capsize=2, markerfacecolor='none',
                    label='Surrogate', markersize=3, alpha=alpha)
        ax[0].errorbar(data['y'] * 1000, ym_prior*1000, xerr=xerr, fmt='ok', capsize=2,
                    label='Model', markersize=3, alpha=alpha)
        ax[0].plot(np.sort(data['y'] * 1000), np.sort(data['y'] * 1000), '-k', alpha=0.6, linewidth=1.2)
        uq.ax_default(ax[0], 'Experimental thrust (mN)', 'Predicted thrust (mN)', legend=True)
        ax[0].set_title('Prior predictive')
        p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        yerr = np.vstack((med - p5, p95 - med))
        ax[1].errorbar(data['y'] * 1000, med, yerr=yerr, xerr=xerr, fmt='or', capsize=2, markerfacecolor='none',
                       label='Surrogate', markersize=3, alpha=alpha)
        ax[1].errorbar(data['y'] * 1000, ym_post * 1000, xerr=xerr, fmt='ok', capsize=2,
                       label='Model', markersize=3, alpha=alpha)
        ax[1].plot(np.sort(data['y'] * 1000), np.sort(data['y'] * 1000), '-k', alpha=0.6, linewidth=1.2)
        uq.ax_default(ax[1], 'Experimental thrust (mN)', '', legend=True)
        ax[1].set_title('Posterior predictive')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-thrust-test.png', dpi=300, format='png')
        plt.show()

        print('-------------Thrust test set relative L2 scores----------------')
        print(f'{"Case":>20} {"Prior":>15} {"Posterior":>15}')
        print(f'{"Surr-Model":>20} {relative_l2(np.percentile(ys_prior, 50, axis=0), ym_prior):15.3f} '
              f'{relative_l2(np.percentile(ys_post, 50, axis=0), ym_post):>15.3f}')
        print(f'{"Surr-Data":>20} {relative_l2(np.percentile(ys_prior, 50, axis=0), data["y"]):15.3f} '
              f'{relative_l2(np.percentile(ys_post, 50, axis=0), data["y"]):>15.3f}')
        print(f'{"Model-Data":>20} {relative_l2(ym_prior, data["y"]):15.3f} '
              f'{relative_l2(ym_post, data["y"]):>15.3f}')

        # Discharge current (Sankovic)
        fig, ax = plt.subplots(1, 2, sharey='row')
        data = DATA['I_D'][0]
        ys_post = fd['thrust-test/posterior/ysurr'][..., 1]
        ys_prior = fd['thrust-test/prior/ysurr'][..., 1]
        ym_post = fd['thrust-test/posterior/ymodel'][..., 1]
        ym_prior = fd['thrust-test/prior/ymodel'][..., 1]
        p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        yerr = np.vstack((med - p5, p95 - med))
        ax[0].errorbar(data['y'], med, yerr=yerr, fmt='or', capsize=2,
                       markerfacecolor='none', label='Surrogate', markersize=3, alpha=alpha)
        ax[0].errorbar(data['y'], ym_prior, fmt='ok', capsize=2,
                       label='Model', markersize=3, alpha=alpha)
        ax[0].plot(np.sort(data['y']), np.sort(data['y']), '-k', alpha=0.6, linewidth=1.2)
        uq.ax_default(ax[0], 'Experimental discharge current (A)', 'Predicted discharge current (A)', legend=True)
        ax[0].set_title('Prior predictive')
        p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        yerr = np.vstack((med - p5, p95 - med))
        ax[1].errorbar(data['y'], med, yerr=yerr, fmt='or', capsize=2,
                       markerfacecolor='none', label='Surrogate', markersize=3, alpha=alpha)
        ax[1].errorbar(data['y'], ym_post, fmt='ok', capsize=2,
                       label='Model', markersize=3, alpha=alpha)
        ax[1].plot(np.sort(data['y']), np.sort(data['y']), '-k', alpha=0.6, linewidth=1.2)
        uq.ax_default(ax[1], 'Experimental discharge current (A)', '', legend=True)
        ax[1].set_title('Posterior predictive')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-discharge-test.png', dpi=300, format='png')
        plt.show()

        print('-------------Discharge current test set relative L2 scores----------------')
        print(f'{"Case":>20} {"Prior":>15} {"Posterior":>15}')
        print(f'{"Surr-Model":>20} {relative_l2(np.percentile(ys_prior, 50, axis=0), ym_prior):15.3f} '
              f'{relative_l2(np.percentile(ys_post, 50, axis=0), ym_post):>15.3f}')
        print(f'{"Surr-Data":>20} {relative_l2(np.percentile(ys_prior, 50, axis=0), data["y"]):15.3f} '
              f'{relative_l2(np.percentile(ys_post, 50, axis=0), data["y"]):>15.3f}')
        print(f'{"Model-Data":>20} {relative_l2(ym_prior, data["y"]):15.3f} '
              f'{relative_l2(ym_post, data["y"]):>15.3f}')

        # Ion velocity (Macdonald)
        data = DATA['uion'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        Nx = data['x'].shape[0]
        z, ys_post = uion_reconstruct(np.array(fd['uion/posterior/ysurr'][..., :-1]))
        z, ys_prior = uion_reconstruct(np.array(fd['uion/prior/ysurr'][..., :-1]))
        ym_post = fd['uion/posterior/ymodel'][..., :-1]
        ym_prior = fd['uion/prior/ymodel'][..., :-1]
        fig, ax = plt.subplots(1, 2, sharey='row')
        ax[0].grid()
        ax[1].grid()
        colors = ['r', 'g', 'b']
        handles, labels = [], []
        for i in range(Nx):
            yerr = 2 * np.sqrt(data['var_y'][i, :])
            ax[0].errorbar(data['loc'] * 1000, data['y'][i, :], yerr=yerr, fmt='o', color=colors[i], capsize=3,
                           markerfacecolor='none', markersize=4)
            h = ax[1].errorbar(data['loc'] * 1000, data['y'][i, :], yerr=yerr, fmt='o', color=colors[i], capsize=3,
                               markerfacecolor='none', markersize=4)
            handles.append(h)
            labels.append(f'{10 ** pb[i]:.2E} Torr')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            ax[0].fill_between(z*1000, p5[i, :], p95[i, :], alpha=alpha, edgecolor=colors[i], facecolor=colors[i])
            ax[0].plot(z*1000, med[i, :], ls='--', c=colors[i], alpha=0.6)
            ax[0].plot(z*1000, ym_prior[i, :], ls='-', c=colors[i], alpha=0.6)
            p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            ax[1].fill_between(z*1000, p5[i, :], p95[i, :], alpha=alpha, edgecolor=colors[i], facecolor=colors[i])
            ax[1].plot(z * 1000, med[i, :], ls='--', c=colors[i], alpha=0.6)
            ax[1].plot(z * 1000, ym_post[i, :], ls='-', c=colors[i], alpha=0.6)
        h1, = ax[1].plot(np.nan, np.nan, ls='--', color=gray, alpha=0.6)
        h2 = ax[1].fill_between(z*1000, np.nan, np.nan, alpha=alpha, edgecolor=gray, facecolor=gray)
        h3, = ax[1].plot(np.nan, np.nan, ls='-', color=gray, alpha=0.6)
        h4 = ax[1].errorbar(np.nan, np.nan, yerr=0, fmt='o', color=gray, capsize=3, markerfacecolor='none', markersize=4)
        uq.ax_default(ax[0], 'Axial distance from anode (mm)', 'Axial ion velocity (m/s)', legend=False)
        uq.ax_default(ax[1], 'Axial distance from anode (mm)', '', legend=False)
        leg = ax[1].legend(handles + [(h2, h1), h3, h4], labels + ['Surrogate', 'Model', 'Experiment'],
                           fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        ax[0].set_title('Prior predictive')
        ax[1].set_title('Posterior predictive')
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-uion.png', dpi=300, format='png')
        plt.show()

        # Ion current density
        data = DATA['jion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        alpha_g = data['loc'][:, 1]
        print(f'Ion current density results given at PB={10**pb[0]:.2E} Torr')
        cov = np.expand_dims(data['var_y'], axis=(-1, -2))
        alpha_i, ys_post = jion_reconstruct(np.array(fd['jion/posterior/ysurr'][..., :-1]), alpha=alpha_g)
        ys_post_pred = np.squeeze(uq.normal_sample(ys_post[..., np.newaxis], cov), axis=-1)
        alpha_i, ys_prior = jion_reconstruct(np.array(fd['jion/prior/ysurr'][..., :-1]), alpha=alpha_g)
        ym_post = fd['jion/posterior/ymodel'][..., :-1]
        ym_prior = fd['jion/prior/ymodel'][..., :-1]
        alpha_i = alpha_i*180/np.pi
        yerr = 2 * np.sqrt(data['var_y'][0, :])
        fig, ax = plt.subplots(1, 2, sharey='row')
        p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
        h1 = ax[0].fill_between(alpha_i, p5[0, :], p95[0, :], alpha=alpha, edgecolor=gray, facecolor=gray)
        h2, = ax[0].plot(alpha_i, med[0, :], ls='--', c='k')
        h3, = ax[0].plot(alpha_i, ym_prior[0, :], ls='-', c='k')
        h4 = ax[0].errorbar(alpha_i, data['y'][0, :], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        ax[0].set_title('Prior predictive')
        uq.ax_default(ax[0], 'Angle from thruster centerline (deg)', r'Ion current density (A/$\text{m}^2$)', legend=False)
        ax[0].set_yscale('log')
        leg = ax[0].legend([(h1, h2), h3, h4], ['Surrogate', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        p5_pred, p95_pred = list(map(lambda x: np.percentile(ys_post_pred, x, axis=0), (5, 95)))
        p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
        h1 = ax[1].fill_between(alpha_i, p95[0, :], p95_pred[0, :], alpha=alpha, edgecolor='b', facecolor='b')
        h2 = ax[1].fill_between(alpha_i, p5_pred[0, :], p5[0, :], alpha=alpha, edgecolor='b', facecolor='b')
        h3 = ax[1].fill_between(alpha_i, p5[0, :], p95[0, :], alpha=alpha, edgecolor='r', facecolor='r')
        h4, = ax[1].plot(alpha_i, med[0, :], ls='--', c='r')
        h5, = ax[1].plot(alpha_i, ym_post[0, :], ls='-', c='r')
        h6 = ax[1].errorbar(alpha_i, data['y'][0, :], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
        ax[1].set_title('Posterior predictive')
        uq.ax_default(ax[1], 'Angle from thruster centerline (deg)', '', legend=False)
        leg = ax[1].legend([(h3, h4), h1, h5, h6], ['Surrogate', 'Surrogate w/noise', 'Model', 'Experiment'],
                           loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        leg.get_frame().set_edgecolor('k')
        ax[1].set_yscale('log')
        ax[0].grid()
        ax[1].grid()
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-jion.png', dpi=300, format='png')
        plt.show()

        fig, ax = plt.subplots(1, 2, sharey='row')
        colors = plt.get_cmap('jet')(np.linspace(0, 1, Nx))
        for i in range(Nx):
            ax[0].plot(alpha_i, data['y'][i, :], '-o', color=colors[i], alpha=0.5, ms=3)
            ax[1].plot(alpha_i, ym_post[i, :], '-', color=colors[i], alpha=0.5)
            ax[1].plot(alpha_i, med[i, :], '--', color=colors[i], alpha=0.5)
            ax[1].plot(np.nan, np.nan, '-', color=colors[i], label=f'{10 ** pb[i]:.2E} Torr')
        ax[1].plot(np.nan, np.nan, '-', color=gray, alpha=0.5, label='Model')
        ax[1].plot(np.nan, np.nan, '--', color=gray, alpha=0.5, label='Surrogate')
        ax[0].set_yscale('log')
        ax[0].grid()
        ax[0].set_title('Experiment')
        uq.ax_default(ax[0], 'Angle from thruster centerline (deg)', r'Ion current density (A/$\text{m}^2$)', legend=False)
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].set_title('Posterior predictive')
        uq.ax_default(ax[1], 'Angle from thruster centerline (deg)', '', legend=True)
        fig.set_size_inches(10, 5)
        fig.tight_layout(w_pad=2)
        fig.savefig('mc-jion-model.png', dpi=300, format='png')
        plt.show()


def extra_plots():
    pass
    # Thrust (Diamant)
    # data = DATA['T'][0]
    # pb = data['x'][:, 0]
    # idx = np.argsort(pb)
    # ys = fd['thrust/ys'][..., 0]
    # I_D = fd['thrust/ys'][..., 1]
    # ymodel = np.array(fd['thrust/ymodel'])[..., 0]
    # ID_model = np.array(fd['thrust/ymodel'])[..., 1]
    # fig, ax = plt.subplots(1, 3)
    # yerr = 2 * np.sqrt(data['var_y'])
    # ax[0].errorbar(10 ** pb, data['y'] * 1000, yerr=yerr * 1000, fmt='o', c=(0.4, 0.4, 0.4), capsize=3,
    #                markerfacecolor='none', label='Experiment', markersize=4)
    # p5 = np.percentile(ys, 5, axis=0) * 1000
    # med = np.percentile(ys, 50, axis=0) * 1000
    # p95 = np.percentile(ys, 95, axis=0) * 1000
    # ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=0.2, edgecolor=(0.4, 0.4, 0.4), facecolor='r')
    # ax[0].plot(10 ** pb[idx], med[idx], '--r', label='Surrogate')
    # ax[0].plot(10 ** pb[idx], ymodel[idx] * 1000, '-k', label='Model')
    # ax[0].grid()
    # xline = 10 ** (np.ones(5) * pb[idx][-5])
    # yline = np.linspace(77.5, 85, 5)
    # ax[0].plot(xline, yline, '-.b', linewidth=1.2, alpha=0.6)
    # xy = (xline[0] - 0.1e-5, yline[0])
    # ax[0].annotate(f'A', xy, xytext=(xy[0] + 1e-5, xy[1] - 0.1), weight='bold',
    #                arrowprops={'arrowstyle': '<|-', 'linewidth': 1.2, 'alpha': 0.6, 'color': 'b'})
    # xy = (xline[0] - 0.1e-5, yline[-1])
    # ax[0].annotate(f'A', xy, xytext=(xy[0] + 1e-5, xy[1] - 0.1), weight='bold',
    #                arrowprops={'arrowstyle': '<|-', 'linewidth': 1.2, 'alpha': 0.6, 'color': 'b'})
    # uq.ax_default(ax[0], 'Background pressure (Torr)', 'Thrust (mN)', legend=False)
    # leg = ax[0].legend(loc='upper right', fancybox=True, facecolor='white', framealpha=1)
    # frame = leg.get_frame()
    # frame.set_edgecolor('k')
    # ax[0].set_xscale('log')
    # ax[1].set_title(r'Section A-A at $P_B=3.81\text{E-5}$ Torr')
    # ax[1].set_xlim((yline[0], yline[-1]))
    # ax[1].hist(ys[:, idx[-5]] * 1000, bins=15, density=True, facecolor='r', alpha=0.2, edgecolor=(0.4, 0.4, 0.4),
    #            lw=1.4, label='Surrogate')
    # ax[1].axvline(ymodel[idx[-5]] * 1000, c='k', ls='-', label='Model')
    # exp_y = data['y'][idx[-5]] * 1000
    # std = yerr[idx[-5]] * 1000 / 2
    # x = np.linspace(exp_y - 3.5 * std, exp_y + 3.5 * std, 100)
    # y = np.squeeze(uq.normal_pdf(x[..., np.newaxis], exp_y, std ** 2))
    # ax[1].plot(x, y, ls='-', c=(0.4, 0.4, 0.4))
    # ax[1].fill_between(x, y1=y, y2=0, lw=0, alpha=0.2, facecolor=(0.8, 0.8, 0.8), label='Experiment')
    # uq.ax_default(ax[1], 'Thrust (mN)', 'PDF', legend=True)
    # ys = np.array(fd['thrust-uq/ys'][..., 0]) * 1000  # (100, 1000), CDF plots at PB=3.81E-5 Torr
    # # I_D = np.array(fd['thrust-uq/ys'][..., 1])
    # ys = np.sort(ys, axis=-1)
    # cdfs = np.arange(1, ys.shape[-1] + 1) / ys.shape[-1]
    # cycol = cycle('bgrcmk')
    # for i in range(0, ys.shape[0], 3):
    #     ax[2].plot(ys[i, :], cdfs, ls='-', c=next(cycol), alpha=0.2)
    # ax[2].set_title(r'Uncertainty at $P_B=3.81\text{E-5}$ Torr')
    # xy = (79.58, 0.2)
    # ax[2].annotate('Epistemic uncertainty', xy, xytext=(xy[0] + 0.27, xy[1]),
    #                arrowprops={'arrowstyle': '<|-|>', 'linewidth': 1.4, 'color': 'k'})
    # uq.ax_default(ax[2], 'Thrust (mN)', 'Aleatoric CDF', legend=False)
    # fig.set_size_inches(10, 6)
    # fig.tight_layout()
    # # fig.savefig('mc-thrust.png', dpi=300, format='png')
    # plt.show()

    # Plot alternate predictions when mdot_a = constant
    # fig, ax = plt.subplots(1, 2)
    # ys_post = fd['thrust/alternate/ysurr'][..., 0]
    # ys_post_pred = np.squeeze(uq.normal_sample(np.array(ys_post)[..., np.newaxis], cov), axis=-1)
    # ym_post = fd['thrust/alternate/ymodel'][..., 0]
    # p5_pred, p95_pred = list(map(lambda x: 1000 * np.percentile(ys_post_pred, x, axis=0), (5, 95)))
    # p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_post, x, axis=0), (5, 50, 95)))
    # h1 = ax[0].fill_between(10 ** pb[idx], p95[idx], p95_pred[idx], alpha=alpha, edgecolor='b', facecolor='b')
    # h2 = ax[0].fill_between(10 ** pb[idx], p5_pred[idx], p5[idx], alpha=alpha, edgecolor='b', facecolor='b')
    # h3 = ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
    # h4, = ax[0].plot(10 ** pb[idx], med[idx], ls='--', c='r')
    # h5, = ax[0].plot(10 ** pb[idx], ym_post[idx] * 1000, ls='-', c='r')
    # h6 = ax[0].errorbar(10 ** pb, data['y'] * 1000, yerr=yerr * 1000, fmt='ok', capsize=3, markerfacecolor='none',
    #                     markersize=4)
    # uq.ax_default(ax[0], 'Background pressure (Torr)', 'Thrust (mN)', legend=False)
    # leg = ax[0].legend([(h3, h4), h1, h5, h6], ['Surrogate', 'Surrogate w/noise', 'Model', 'Experiment'],
    #                    loc='upper right', fancybox=True, facecolor='white', framealpha=1)
    # leg.get_frame().set_edgecolor('k')
    # ax[0].set_xscale('log')
    # ax[0].set_ylim(top=120)
    # ys_post = fd['thrust/alternate/ysurr'][..., 1]
    # ym_post = fd['thrust/alternate/ymodel'][..., 1]
    # p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
    # h1 = ax[1].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
    # h2, = ax[1].plot(10 ** pb[idx], med[idx], ls='--', c='r')
    # h3, = ax[1].plot(10 ** pb[idx], ym_post[idx], ls='-', c='r')
    # h4, = ax[1].plot(10 ** pb, np.ones(pb.shape[0])*DISCHARGE_CURRENT, '-ok', markersize=4)
    # uq.ax_default(ax[1], 'Background pressure (Torr)', 'Discharge current (A)', legend=False)
    # leg = ax[1].legend([(h1, h2), h3, h4], ['Surrogate', 'Model', 'Experiment'],
    #                    loc='upper right', fancybox=True, facecolor='white', framealpha=1)
    # leg.get_frame().set_edgecolor('k')
    # ax[1].set_xscale('log')
    # ax[1].set_ylim(top=10, bottom=3.5)
    # fig.set_size_inches(10, 5)
    # fig.tight_layout(w_pad=2)
    # fig.savefig('mc-thrust-alternate.png', dpi=300, format='png')
    # plt.show()

    # Divergence angle
    # idx = np.logical_and(alpha_i >= 0, alpha_i <= 90)
    # alpha_rad = alpha_i[idx] * np.pi/180
    # num_int = ymodel[:, idx] * np.cos(alpha_rad) * np.sin(alpha_rad)
    # den_int = ymodel[:, idx] * np.cos(alpha_rad)
    # num_int_exp = data['y'][:, idx] * np.cos(alpha_rad) * np.sin(alpha_rad)
    # den_int_exp = data['y'][:, idx] * np.cos(alpha_rad)
    # with np.errstate(divide='ignore'):
    #     cos_div = simps(num_int, alpha_rad, axis=-1) / simps(den_int, alpha_rad, axis=-1)
    #     cos_div[cos_div == np.inf] = np.nan
    #     cos_div_exp = simps(num_int_exp, alpha_rad, axis=-1) / simps(den_int_exp, alpha_rad, axis=-1)
    #     cos_div_exp[cos_div_exp == np.inf] = np.nan
    #
    # fig, ax = plt.subplots()
    # ax.plot(10 ** pb, np.arccos(cos_div_exp)*180/np.pi, '-ko', ms=3, label='Experiment')
    # ax.plot(10 ** pb, np.arccos(cos_div)*180/np.pi, '--k', label='Model')
    # ax.grid()
    # ax.set_xscale('log')
    # uq.ax_default(ax, 'Background pressure (Torr)', 'Divergence angle (deg)', legend=True)
    # fig.set_size_inches(7, 6)
    # fig.tight_layout()
    # # fig.savefig('mc-jion-div.png', dpi=300, format='png')
    # plt.show()

    # fig, ax = plt.subplots(2, int(Nx/2), sharey='row', sharex='col')
    # idx = 0
    # for i in range(2):
    #     for j in range(int(Nx/2)):
    #         yerr = 2 * np.sqrt(data['var_y'][idx, :])
    #         ax[i, j].errorbar(alpha_g*180/np.pi, data['y'][idx, :], yerr=yerr, fmt='or', capsize=3,
    #                           markerfacecolor='none', label='Experiment', markersize=4)
    #         xlabel = 'Angle from thruster centerline (deg)' if i == 1 else ''
    #         ylabel = 'Ion current density ($A/m^2$)' if j == 0 else ''
    #         legend = i == 0 and j == int(Nx/2) - 1
    #         p5 = np.percentile(jion_i[:, idx, :], 5, axis=0)
    #         med = np.percentile(jion_i[:, idx, :], 50, axis=0)
    #         p95 = np.percentile(jion_i[:, idx, :], 95, axis=0)
    #         ax[i, j].fill_between(alpha_i, p5, p95, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    #         ax[i, j].plot(alpha_i, med, '-k', label='Surrogate')
    #         ax[i, j].plot(alpha_i, ymodel[idx, :], '--k', label='Model')
    #         uq.ax_default(ax[i, j], xlabel, ylabel, legend=legend)
    #         ax[i, j].set_title(f'{10 ** pb[idx]:.2E} Torr')
    #         ax[i, j].set_yscale('log')
    #         idx += 1
    # fig.set_size_inches(3*int(Nx/2), 6)
    # fig.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # run_models()
    spt100_monte_carlo()
