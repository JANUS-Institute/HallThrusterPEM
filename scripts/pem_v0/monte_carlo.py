"""Script for surrogate-enabled Monte Carlo forward UQ analysis.

I do not seriously intend that anyone will understand how this specific script works, only that it does.
The general idea is to gather all the Monte Carlo samples, save them in an .h5 file, then run a function
that makes all of the plots/tables used in the journal paper. Good luck.
"""
from pathlib import Path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import uqtils as uq
from scipy.interpolate import interp1d
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
        nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), use_model='best')
        ymodel_surr = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING)
        fd.create_dataset('vcc/posterior/xsurr', data=xs)
        fd.create_dataset('vcc/posterior/ysurr', data=ys)
        fd.create_dataset('vcc/posterior/xmodel', data=xmodel)
        fd.create_dataset('vcc/posterior/ymodel', data=ymodel)
        fd.create_dataset('vcc/posterior/ymodel_surr', data=ymodel_surr)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING), axis=-1)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1]) / 2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), use_model='best')
        ymodel_surr = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING)
        fd.create_dataset('vcc/prior/xsurr', data=xs)
        fd.create_dataset('vcc/prior/ysurr', data=ys)
        fd.create_dataset('vcc/prior/xmodel', data=xmodel)
        fd.create_dataset('vcc/prior/ymodel', data=ymodel)
        fd.create_dataset('vcc/prior/ymodel_surr', data=ymodel_surr)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], training=TRAINING)
        fd.create_dataset('thrust/posterior/xsurr', data=xs)
        fd.create_dataset('thrust/posterior/ysurr', data=ys)
        fd.create_dataset('thrust/posterior/xmodel', data=xmodel)
        fd.create_dataset('thrust/posterior/ymodel', data=ymodel)
        fd.create_dataset('thrust/posterior/ymodel_surr', data=ymodel_surr)

        # xs[..., 2] = data['x'][idx[0], 2]  # Alternate predictions with constant mdot_a
        # ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        # nominal.update({str(v): v.nominal for i, v in enumerate(THETA_VARS)})
        # xmodel[..., 2] = data['x'][idx[0], 2]
        # ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        # fd.create_dataset('thrust/alternate/xsurr', data=xs)
        # fd.create_dataset('thrust/alternate/ysurr', data=ys)
        # fd.create_dataset('thrust/alternate/xmodel', data=xmodel)
        # fd.create_dataset('thrust/alternate/ymodel', data=ymodel)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1])/2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        ymodel_surr = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], training=TRAINING)
        fd.create_dataset('thrust/prior/xsurr', data=xs)
        fd.create_dataset('thrust/prior/ysurr', data=ys)
        fd.create_dataset('thrust/prior/xmodel', data=xmodel)
        fd.create_dataset('thrust/prior/ymodel', data=ymodel)
        fd.create_dataset('thrust/prior/ymodel_surr', data=ymodel_surr)

        # idx = np.argsort(pb)[-5]
        # nominal = {'PB': pb[idx], 'Va': data['x'][idx, 1], 'mdot_a': data['x'][idx, 2]}
        # xs = np.zeros((100, Ns, len(SURR.x_vars)))
        # for i in range(100):
        #     theta = posterior_sampler(1)
        #     nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        #     xs[i, ...] = SURR.sample_inputs(Ns, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        # ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        # fd.create_dataset('thrust-uq/posterior/xsurr', data=xs)
        # fd.create_dataset('thrust-uq/posterior/ysurr', data=ys)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], training=TRAINING)
        fd.create_dataset('thrust-test/posterior/xsurr', data=xs)
        fd.create_dataset('thrust-test/posterior/ysurr', data=ys)
        fd.create_dataset('thrust-test/posterior/xmodel', data=xmodel)
        fd.create_dataset('thrust-test/posterior/ymodel', data=ymodel)
        fd.create_dataset('thrust-test/posterior/ymodel_surr', data=ymodel_surr)

        theta = prior_sampler(sample_shape)
        nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=['T', 'I_D'], training=TRAINING)
        nominal.update({str(v): v.mu if isinstance(v, NormalRV) else (v.bounds()[0] + v.bounds()[1]) / 2
                        for i, v in enumerate(THETA_VARS)})
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], use_model='best')
        ymodel_surr = SURR.predict(xmodel, qoi_ind=['T', 'I_D'], training=TRAINING)
        fd.create_dataset('thrust-test/prior/xsurr', data=xs)
        fd.create_dataset('thrust-test/prior/ysurr', data=ys)
        fd.create_dataset('thrust-test/prior/xmodel', data=xmodel)
        fd.create_dataset('thrust-test/prior/ymodel', data=ymodel)
        fd.create_dataset('thrust-test/prior/ymodel_surr', data=ymodel_surr)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=qois, training=TRAINING)
        fd.create_dataset('uion/posterior/xsurr', data=xs)
        fd.create_dataset('uion/posterior/ysurr', data=ys)
        fd.create_dataset('uion/posterior/xmodel', data=xmodel)
        fd.create_dataset('uion/posterior/ymodel', data=np.concatenate((ymodel[..., 7:], ymodel[..., 1:2]), axis=-1))
        fd.create_dataset('uion/posterior/ymodel_surr', data=ymodel_surr)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=qois, training=TRAINING)
        fd.create_dataset('uion/prior/xsurr', data=xs)
        fd.create_dataset('uion/prior/ysurr', data=ys)
        fd.create_dataset('uion/prior/xmodel', data=xmodel)
        fd.create_dataset('uion/prior/ymodel', data=np.concatenate((ymodel[..., 7:], ymodel[..., 1:2]), axis=-1))
        fd.create_dataset('uion/prior/ymodel_surr', data=ymodel_surr)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=qois, training=TRAINING)
        fd.create_dataset('jion/posterior/xsurr', data=xs)
        fd.create_dataset('jion/posterior/ysurr', data=ys)
        fd.create_dataset('jion/posterior/xmodel', data=xmodel)
        fd.create_dataset('jion/posterior/ymodel', data=np.concatenate((jion_interp, I_B0[..., 1:]), axis=-1))
        fd.create_dataset('jion/posterior/ymodel_surr', data=ymodel_surr)

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
        ymodel_surr = SURR.predict(xmodel, qoi_ind=qois, training=TRAINING)
        fd.create_dataset('jion/prior/xsurr', data=xs)
        fd.create_dataset('jion/prior/ysurr', data=ys)
        fd.create_dataset('jion/prior/xmodel', data=xmodel)
        fd.create_dataset('jion/prior/ymodel', data=np.concatenate((jion_interp, I_B0[..., 1:]), axis=-1))
        fd.create_dataset('jion/prior/ymodel_surr', data=ymodel_surr)


def relative_l2(pred, targ):
    targ = np.atleast_1d(targ)
    return np.sqrt(np.mean((pred - targ)**2, axis=-1) / np.mean(targ**2, axis=-1))


def print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, exp_data, exp_noise):
    """Print summary validation metrics for journal tables (mean and std of relative l2 error)"""
    avg_func = np.mean
    surr_e_prior = relative_l2(ys_prior, exp_data)  # (M,) model samples
    surr_e_post = relative_l2(ys_post, exp_data)
    model_e_prior = relative_l2(ym_prior, exp_data)
    model_e_post = relative_l2(ym_post, exp_data)
    surr_model_prior = relative_l2(ym_surr_prior, ym_prior)
    surr_model_post = relative_l2(ym_surr_post, ym_post)
    surr_med_e_prior = relative_l2(ym_surr_prior, exp_data)
    surr_med_e_post = relative_l2(ym_surr_post, exp_data)
    print(f'{"Case":>20} {"Prior mu":>8} {"Prior s":>8} {"Prior r":>8} {"Post mu":>8} {"Post s":>8} {"Post r":>8}')
    print(f'{"Surr-Data":>20} {avg_func(surr_e_prior):>8.3f} {np.std(surr_e_prior):>8.3f} '
          f'{avg_func(surr_e_prior) / exp_noise:>8.3f} '
          f'{avg_func(surr_e_post):>8.3f} {np.std(surr_e_post):>8.3f} '
          f'{avg_func(surr_e_post) / exp_noise:>8.3f}')
    print(f'{"Model-Med-Data":>20} {avg_func(model_e_prior):>8.3f} {np.std(model_e_prior):>8.3f} '
          f'{avg_func(model_e_prior) / exp_noise:>8.3f} '
          f'{avg_func(model_e_post):>8.3f} {np.std(model_e_post):>8.3f} '
          f'{avg_func(model_e_post) / exp_noise:>8.3f}')
    print(f'{"Surr-Model":>20} {avg_func(surr_model_prior):>8.3f} {np.std(surr_model_prior):>8.3f} '
          f'{avg_func(surr_model_prior) / exp_noise:>8.3f} '
          f'{avg_func(surr_model_post):>8.3f} {np.std(surr_model_post):>8.3f} '
          f'{avg_func(surr_model_post) / exp_noise:>8.3f}')
    print(f'{"Surr-Med-Model":>20} {avg_func(surr_med_e_prior):>8.3f} {np.std(surr_med_e_prior):>8.3f} '
          f'{avg_func(surr_med_e_prior) / exp_noise:>8.3f} '
          f'{avg_func(surr_med_e_post):>8.3f} {np.std(surr_med_e_post):>8.3f} '
          f'{avg_func(surr_med_e_post) / exp_noise:>8.3f}')


def spt100_monte_carlo(Ns=1000, plot=True):
    """Plot `[V_cc, T, uion, jion]` against SPT-100 experimental data with UQ bounds."""
    file = Path('monte-carlo.h5')
    gray = (0.5, 0.5, 0.5)
    alpha = 0.2
    figsize = (6, 5)
    if not file.is_file():
        run_models(Ns)

    with h5py.File(file, 'r') as fd, plt.style.context("uqtils.default"):
        # Cathode coupling voltage
        data = DATA['V_cc'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        ys_post = fd['vcc/posterior/ysurr']
        ys_prior = fd['vcc/prior/ysurr']
        ym_post = fd['vcc/posterior/ymodel']
        ym_prior = fd['vcc/prior/ymodel']
        ym_surr_post = fd['vcc/posterior/ymodel_surr']
        ym_surr_prior = fd['vcc/prior/ymodel_surr']
        yerr = 2 * np.sqrt(data['var_y'])
        ys_post_pred = np.squeeze(uq.normal_sample(np.array(ys_post)[..., np.newaxis], (np.mean(yerr)/2)**2), axis=-1)

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            prior_h = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
            prior_h2, = ax.plot(10 ** pb[idx], med[idx], ls='-', c='k')
            exp_h = ax.errorbar(10 ** pb, data['y'], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
            ax.set_xscale('log')
            ax.set_xlim(left=1e-6, right=1e-4)
            ax.set_ylim(bottom=0)
            leg = dict(handles=[(prior_h, prior_h2), exp_h], labels=['Model', 'Experiment'],
                       loc='upper left', fancybox=True, facecolor='white', framealpha=1)
            uq.ax_default(ax, 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=leg)
            fig.savefig('mc-vcc-prior.pdf', format='pdf', bbox_inches='tight')
            plt.show()

            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5_pred, p95_pred = list(map(lambda x: np.percentile(ys_post_pred, x, axis=0), (5, 95)))
            p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(10 ** pb[idx], p95[idx], p95_pred[idx], alpha=alpha, edgecolor='b', facecolor='b')
            h2 = ax.fill_between(10 ** pb[idx], p5_pred[idx], p5[idx], alpha=alpha, edgecolor='b', facecolor='b')
            h3 = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
            h4, = ax.plot(10 ** pb[idx], med[idx], ls='-', c='r')
            h5 = ax.errorbar(10 ** pb, data['y'], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
            ax.set_xscale('log')
            ax.set_xlim(left=1e-6, right=1e-4)
            leg = dict(handles=[(h3, h4), (h1, h4), h5], labels=['Model', 'Model w/noise', 'Experiment'],
                       loc='upper left', fancybox=True, facecolor='white', framealpha=1)
            uq.ax_default(ax, 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=leg)
            fig.savefig('mc-vcc-post.pdf', format='pdf', bbox_inches='tight')
            plt.show()

        print('-------------Cathode coupling voltage training set relative L2 scores----------------')
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, data['y'], 0.01)

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
        ym_surr_post = fd['thrust/posterior/ymodel_surr'][..., 0]
        ym_surr_prior = fd['thrust/prior/ymodel_surr'][..., 0]

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
            h2, = ax.plot(10 ** pb[idx], med[idx], ls='--', c='k')
            h3, = ax.plot(10 ** pb[idx], ym_prior[idx]*1000, ls='-', c='k')
            h4 = ax.errorbar(10 ** pb, data['y'] * 1000, yerr=yerr*1000, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
            ax.set_xscale('log')
            ax.set_xlim(left=1e-6, right=1e-4)
            ax.set_ylim(bottom=67, top=129)
            leg = dict(handles=[(h1, h2), h3, h4], labels=['Surrogate', 'Model', 'Experiment'], loc='upper right')
            uq.ax_default(ax, 'Background pressure (Torr)', 'Thrust (mN)', legend=leg)
            fig.savefig('mc-thrust-prior.pdf', format='pdf', bbox_inches='tight')
            plt.show()

            with matplotlib.rc_context(rc={'legend.fontsize': 13}):
                fig, ax = plt.subplots(figsize=figsize, layout='tight')
                p5_pred, p95_pred = list(map(lambda x: 1000 * np.percentile(ys_post_pred, x, axis=0), (5, 95)))
                p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_post, x, axis=0), (5, 50, 95)))
                h1 = ax.fill_between(10 ** pb[idx], p95[idx], p95_pred[idx], alpha=alpha, edgecolor='b', facecolor='b')
                h2 = ax.fill_between(10 ** pb[idx], p5_pred[idx], p5[idx], alpha=alpha, edgecolor='b', facecolor='b')
                h3 = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
                h4, = ax.plot(10 ** pb[idx], med[idx], ls='--', c='r')
                h5, = ax.plot(10 ** pb[idx], ym_post[idx]*1000, ls='-', c='r')
                h6 = ax.errorbar(10 ** pb, data['y'] * 1000, yerr=yerr*1000, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
                leg = dict(handles=[(h3, h4), (h1, h4), h5, h6], labels=['Surrogate', 'Surrogate w/noise', 'Model', 'Experiment'], loc='upper right')
                ax.set_xscale('log')
                ax.set_xlim(left=1e-6, right=1e-4)
                ax.set_ylim(bottom=67, top=129)
                uq.ax_default(ax, 'Background pressure (Torr)', 'Thrust (mN)', legend=leg)
                fig.savefig('mc-thrust-post.pdf', format='pdf', bbox_inches='tight')
                plt.show()

        print('-------------Thrust training set relative L2 scores----------------')
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, data['y'], 0.01)

        # Discharge current (Diamant)
        ys_post = fd['thrust/posterior/ysurr'][..., 1]
        ys_prior = fd['thrust/prior/ysurr'][..., 1]
        ym_post = fd['thrust/posterior/ymodel'][..., 1]
        ym_prior = fd['thrust/prior/ymodel'][..., 1]
        ym_surr_post = fd['thrust/posterior/ymodel_surr'][..., 1]
        ym_surr_prior = fd['thrust/prior/ymodel_surr'][..., 1]

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor=gray, facecolor=gray)
            h2, = ax.plot(10 ** pb[idx], med[idx], ls='--', c='k')
            h3, = ax.plot(10 ** pb[idx], ym_prior[idx], ls='-', c='k')
            h4 = ax.errorbar(10 ** pb, np.ones(pb.shape[0])*DISCHARGE_CURRENT, yerr=0.1*DISCHARGE_CURRENT,
                                fmt='ok', markersize=4, markerfacecolor='none', capsize=3)
            ax.set_xscale('log')
            ax.set_xlim([1e-6, 1e-4])
            ax.set_ylim([0, 35])
            leg = dict(handles=[(h1, h2), h3, h4], labels=['Surrogate', 'Model', 'Experiment'], loc='upper right')
            uq.ax_default(ax, 'Background pressure (Torr)', 'Discharge current (A)', legend=leg)
            fig.savefig('mc-discharge-prior.pdf', format='pdf', bbox_inches='tight')
            plt.show()

            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=alpha, edgecolor='r', facecolor='r')
            h2, = ax.plot(10 ** pb[idx], med[idx], ls='--', c='r')
            h3, = ax.plot(10 ** pb[idx], ym_post[idx], ls='-', c='r')
            h4 = ax.errorbar(10 ** pb, np.ones(pb.shape[0])*DISCHARGE_CURRENT, yerr=0.1*DISCHARGE_CURRENT,
                                fmt='ok', markersize=4, markerfacecolor='none', capsize=3)
            leg = dict(handles=[(h1, h2), h3, h4], labels=['Surrogate', 'Model', 'Experiment'], loc='upper right')
            ax.set_xscale('log')
            ax.set_xlim([1e-6, 1e-4])
            ax.set_ylim([0, 35])
            uq.ax_default(ax, 'Background pressure (Torr)', 'Discharge current (A)', legend=leg)
            fig.savefig('mc-discharge-post.pdf', format='pdf', bbox_inches='tight')
            plt.show()

        print('-------------Discharge current training set relative L2 scores----------------')
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, 4.5, 0.1)

        # Thrust (Sankovic 1993)
        data = DATA['T'][1]
        ys_post = fd['thrust-test/posterior/ysurr'][..., 0]
        ys_prior = fd['thrust-test/prior/ysurr'][..., 0]
        ym_post = fd['thrust-test/posterior/ymodel'][..., 0]
        ym_prior = fd['thrust-test/prior/ymodel'][..., 0]
        ym_surr_post = fd['thrust-test/posterior/ymodel_surr'][..., 0]
        ym_surr_prior = fd['thrust-test/prior/ymodel_surr'][..., 0]

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            xerr = 2 * np.sqrt(data['var_y']) * 1000
            p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            yerr = np.vstack((med - p5, p95 - med))
            ax.errorbar(data['y'] * 1000, med, yerr=yerr, xerr=xerr, fmt='or', capsize=2, markerfacecolor='none',
                        label='Surrogate', markersize=3, alpha=0.4)
            ax.errorbar(data['y'] * 1000, ym_prior*1000, xerr=xerr, fmt='ok', capsize=2,
                        label='Model', markersize=3, alpha=0.4)
            ax.plot(np.sort(data['y'] * 1000), np.sort(data['y'] * 1000), '-k', alpha=0.6, linewidth=1.2)
            uq.ax_default(ax, 'Experimental thrust (mN)', 'Predicted thrust (mN)', legend=False)
            ax.set_ylim([15, 120])
            fig.savefig('mc-thrust-test-prior.pdf', format='pdf', bbox_inches='tight')
            plt.show()

            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: 1000 * np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            yerr = np.vstack((med - p5, p95 - med))
            ax.errorbar(data['y'] * 1000, med, yerr=yerr, xerr=xerr, fmt='or', capsize=2, markerfacecolor='none',
                           label='Surrogate', markersize=3, alpha=0.4)
            ax.errorbar(data['y'] * 1000, ym_post * 1000, xerr=xerr, fmt='ok', capsize=2,
                           label='Model', markersize=3, alpha=0.4)
            ax.plot(np.sort(data['y'] * 1000), np.sort(data['y'] * 1000), '-k', alpha=0.6, linewidth=1.2)
            uq.ax_default(ax, 'Experimental thrust (mN)', 'Predicted thrust (mN)', legend=True)
            ax.set_ylim([15, 120])
            fig.savefig('mc-thrust-test-post.pdf', bbox_inches='tight', format='pdf')
            plt.show()

        print('-------------Thrust test set relative L2 scores----------------')
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, data['y'], 0.01)

        # Discharge current (Sankovic)
        data = DATA['I_D'][0]
        ys_post = fd['thrust-test/posterior/ysurr'][..., 1]
        ys_prior = fd['thrust-test/prior/ysurr'][..., 1]
        ym_post = fd['thrust-test/posterior/ymodel'][..., 1]
        ym_prior = fd['thrust-test/prior/ymodel'][..., 1]
        ym_surr_post = fd['thrust-test/posterior/ymodel_surr'][..., 1]
        ym_surr_prior = fd['thrust-test/prior/ymodel_surr'][..., 1]

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            yerr = np.vstack((med - p5, p95 - med))
            ax.errorbar(data['y'], med, yerr=yerr, fmt='or', capsize=2,
                           markerfacecolor='none', label='Surrogate', markersize=3, alpha=0.4)
            ax.errorbar(data['y'], ym_prior, fmt='ok', capsize=2,
                           label='Model', markersize=3, alpha=0.4)
            ax.plot(np.sort(data['y']), np.sort(data['y']), '-k', alpha=0.6, linewidth=1.2)
            uq.ax_default(ax, 'Experimental discharge current (A)', 'Predicted discharge current (A)', legend=False)
            ax.set_ylim([1, 34])
            fig.savefig('mc-discharge-test-prior.pdf', bbox_inches='tight', format='pdf')
            plt.show()

            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            yerr = np.vstack((med - p5, p95 - med))
            ax.errorbar(data['y'], med, yerr=yerr, fmt='or', capsize=2,
                           markerfacecolor='none', label='Surrogate', markersize=3, alpha=0.4)
            ax.errorbar(data['y'], ym_post, fmt='ok', capsize=2,
                           label='Model', markersize=3, alpha=0.4)
            ax.plot(np.sort(data['y']), np.sort(data['y']), '-k', alpha=0.6, linewidth=1.2)
            uq.ax_default(ax, 'Experimental discharge current (A)', 'Predicted discharge current (A)', legend=True)
            ax.set_ylim([1, 34])
            fig.savefig('mc-discharge-test-post.pdf', bbox_inches='tight', format='pdf')
            plt.show()

        print('-------------Discharge current test set relative L2 scores----------------')
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, data['y'], 0.1)

        # Ion velocity (Macdonald)
        data = DATA['uion'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        Nx = data['x'].shape[0]
        z, ys_post = uion_reconstruct(np.array(fd['uion/posterior/ysurr'][..., :-1]))
        z, ys_prior = uion_reconstruct(np.array(fd['uion/prior/ysurr'][..., :-1]))
        ym_post = fd['uion/posterior/ymodel'][..., :-1]
        ym_prior = fd['uion/prior/ymodel'][..., :-1]
        z, ym_surr_post = uion_reconstruct(np.array(fd['uion/posterior/ymodel_surr'][..., :-1]))
        z, ym_surr_prior = uion_reconstruct(np.array(fd['uion/prior/ymodel_surr'][..., :-1]))
        zdata = data['loc']

        if plot:
            fig1, ax1 = plt.subplots(figsize=figsize, layout='tight')
            fig2, ax2 = plt.subplots(figsize=figsize, layout='tight')
            ax = [ax1, ax2]
            colors = ['r', 'g', 'b']
            handles, labels = [], []
            for i in range(Nx):
                yerr = 2 * np.sqrt(data['var_y'][i, :]) / 1000
                ax[0].errorbar(data['loc'] * 1000, data['y'][i, :]/1000, yerr=yerr, fmt='o', color=colors[i], capsize=3,
                               markerfacecolor='none', markersize=4)
                h = ax[1].errorbar(data['loc'] * 1000, data['y'][i, :]/1000, yerr=yerr, fmt='o', color=colors[i], capsize=3,
                                   markerfacecolor='none', markersize=4)
                handles.append(h)
                labels.append(f'{10 ** pb[i]:.2E} Torr')
                p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0)/1000, (5, 50, 95)))
                ax[0].fill_between(z*1000, p5[i, :], p95[i, :], alpha=alpha, edgecolor=colors[i], facecolor=colors[i])
                ax[0].plot(z*1000, med[i, :], ls='--', c=colors[i], alpha=0.6)
                ax[0].plot(z*1000, ym_prior[i, :]/1000, ls='-', c=colors[i])
                p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0)/1000, (5, 50, 95)))
                ax[1].fill_between(z*1000, p5[i, :], p95[i, :], alpha=alpha, edgecolor=colors[i], facecolor=colors[i])
                ax[1].plot(z * 1000, med[i, :], ls='--', c=colors[i], alpha=0.6)
                ax[1].plot(z * 1000, ym_post[i, :]/1000, ls='-', c=colors[i])
            h1, = ax[1].plot(np.nan, np.nan, ls='--', color=gray, alpha=0.6)
            h2 = ax[1].fill_between(z*1000, np.nan, np.nan, alpha=alpha, edgecolor=gray, facecolor=gray)
            h3, = ax[1].plot(np.nan, np.nan, ls='-', color=gray)
            h4 = ax[1].errorbar(np.nan, np.nan, yerr=0, fmt='o', color=gray, capsize=3, markerfacecolor='none', markersize=4)
            leg = dict(handles=handles + [(h2, h1), h3, h4], labels=labels + ['Surrogate', 'Model', 'Experiment'])
            uq.ax_default(ax[0], 'Axial distance from anode (mm)', 'Axial ion velocity (km/s)', legend=False)
            uq.ax_default(ax[1], 'Axial distance from anode (mm)', 'Axial ion velocity (km/s)', legend=leg)
            ax[0].set_ylim([-4, 22])
            ax[1].set_ylim([-4, 22])
            fig1.savefig('mc-uion-prior.pdf', bbox_inches='tight', format='pdf')
            fig2.savefig('mc-uion-post.pdf', bbox_inches='tight', format='pdf')
            plt.show()

        print('-------------Ion velocity training set relative L2 scores----------------')
        Ns = ys_prior.shape[0]
        exp_data = np.ravel(data['y'])                                          # (Ndata, )
        ys_prior = interp1d(z, ys_prior, axis=-1)(zdata).reshape((Ns, -1))      # (Ns, Ndata)
        ys_post = interp1d(z, ys_post, axis=-1)(zdata).reshape((Ns, -1))        # (Ns, Ndata)
        ym_prior = np.ravel(interp1d(z, ym_prior, axis=-1)(zdata))              # (Ndata, )
        ym_post = np.ravel(interp1d(z, ym_post, axis=-1)(zdata))                # (Ndata, )
        ym_surr_prior = np.ravel(interp1d(z, ym_surr_prior, axis=-1)(zdata))    # (Ndata, )
        ym_surr_post = np.ravel(interp1d(z, ym_surr_post, axis=-1)(zdata))      # (Ndata, )
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, exp_data, 0.05)

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
        alpha_i, ym_surr_prior = jion_reconstruct(np.array(fd['jion/prior/ymodel_surr'][..., :-1]), alpha=alpha_g)
        alpha_i, ym_surr_post = jion_reconstruct(np.array(fd['jion/posterior/ymodel_surr'][..., :-1]), alpha=alpha_g)
        ym_post = fd['jion/posterior/ymodel'][..., :-1]
        ym_prior = fd['jion/prior/ymodel'][..., :-1]
        alpha_i = alpha_i*180/np.pi
        yerr = 2 * np.sqrt(data['var_y'][0, :])

        if plot:
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5, med, p95 = list(map(lambda x: np.percentile(ys_prior, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(alpha_i, p5[0, :], p95[0, :], alpha=alpha, edgecolor=gray, facecolor=gray)
            h2, = ax.plot(alpha_i, med[0, :], ls='--', c='k')
            h3, = ax.plot(alpha_i, ym_prior[0, :], ls='-', c='k')
            h4 = ax.errorbar(alpha_i, data['y'][0, :], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
            ax.set_yscale('log')
            ax.set_ylim([1e-3, 18])
            leg = dict(handles=[(h1, h2), h3, h4], labels=['Surrogate', 'Model', 'Experiment'], borderaxespad=0.7, loc='lower left')
            uq.ax_default(ax, 'Angle from thruster centerline (deg)', r'Ion current density (A/$\mathrm{m}^2$)', legend=leg)
            fig.savefig('mc-jion-prior.pdf', format='pdf', bbox_inches='tight')
            plt.show()

            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            p5_pred, p95_pred = list(map(lambda x: np.percentile(ys_post_pred, x, axis=0), (5, 95)))
            p5, med, p95 = list(map(lambda x: np.percentile(ys_post, x, axis=0), (5, 50, 95)))
            h1 = ax.fill_between(alpha_i, p95[0, :], p95_pred[0, :], alpha=alpha, edgecolor='b', facecolor='b')
            h2 = ax.fill_between(alpha_i, p5_pred[0, :], p5[0, :], alpha=alpha, edgecolor='b', facecolor='b')
            h3 = ax.fill_between(alpha_i, p5[0, :], p95[0, :], alpha=alpha, edgecolor='r', facecolor='r')
            h4, = ax.plot(alpha_i, med[0, :], ls='--', c='r')
            h5, = ax.plot(alpha_i, ym_post[0, :], ls='-', c='r')
            h6 = ax.errorbar(alpha_i, data['y'][0, :], yerr=yerr, fmt='ok', capsize=3, markerfacecolor='none', markersize=4)
            leg = dict(handles=[(h3, h4), (h1, h4), h5, h6], labels=['Surrogate', 'Surrogate w/noise', 'Model', 'Experiment'],
                       loc='lower left', borderaxespad=0.7)
            ax.set_yscale('log')
            ax.set_ylim([1e-3, 18])
            uq.ax_default(ax, 'Angle from thruster centerline (deg)', r'Ion current density (A/$\mathrm{m}^2$)', legend=leg)
            fig.savefig('mc-jion-post.pdf', bbox_inches='tight', format='pdf')
            plt.show()

            with matplotlib.rc_context(rc={'legend.fontsize': 13}):
                fig1, ax1 = plt.subplots(figsize=figsize, layout='tight')
                fig2, ax2 = plt.subplots(figsize=figsize, layout='tight')
                ax = [ax1, ax2]
                colors = plt.get_cmap('jet')(np.linspace(0, 1, Nx))
                for i in range(Nx):
                    ax[0].plot(alpha_i, data['y'][i, :], '-o', color=colors[i], alpha=0.5, ms=3)
                    ax[1].plot(alpha_i, ym_post[i, :], '-', color=colors[i], alpha=0.5)
                    ax[1].plot(alpha_i, med[i, :], '--', color=colors[i], alpha=0.5)
                    ax[1].plot(np.nan, np.nan, '-', color=colors[i], label=f'{10 ** pb[i]:.2E} Torr')
                ax[1].plot(np.nan, np.nan, '-', color=gray, alpha=0.5, label='Model')
                ax[1].plot(np.nan, np.nan, '--', color=gray, alpha=0.5, label='Surrogate')
                ax[0].set_yscale('log')
                ax[0].set_ylim([1e-3, 18])
                ax[1].set_yscale('log')
                ax[1].set_ylim([1e-3, 18])
                leg = dict(loc='lower left', borderaxespad=0.7)
                uq.ax_default(ax[0], 'Angle from thruster centerline (deg)', r'Ion current density (A/$\mathrm{m}^2$)', legend=False)
                uq.ax_default(ax[1], 'Angle from thruster centerline (deg)', r'Ion current density (A/$\mathrm{m}^2$)', legend=leg)
                fig1.savefig('mc-jion-exp.pdf', bbox_inches='tight', format='pdf')
                fig2.savefig('mc-jion-model.pdf', bbox_inches='tight', format='pdf')
                plt.show()

        print('-------------Ion current density training set relative L2 scores----------------')
        Ns = ys_prior.shape[0]
        exp_data = np.ravel(data['y'])              # (Ndata, )
        ys_prior = ys_prior.reshape((Ns, -1))       # (Ns, Ndata)
        ys_post = ys_post.reshape((Ns, -1))         # (Ns, Ndata)
        ym_prior = np.ravel(ym_prior)               # (Ndata, )
        ym_post = np.ravel(ym_post)                 # (Ndata, )
        ym_surr_prior = np.ravel(ym_surr_prior)     # (Ndata, )
        ym_surr_post = np.ravel(ym_surr_post)       # (Ndata, )
        print_l2_error(ys_prior, ys_post, ym_prior, ym_post, ym_surr_prior, ym_surr_post, exp_data, 0.2)


def get_allocation(surr):
    """Return total cumulative cost during training (including offline cost)"""
    cost_cum = []  # Cumulative cost allocation during training
    comp = surr['Thruster']
    index_set = []
    candidate_set = []

    def activate_index(alpha, beta):
        # Add all possible new candidates (distance of one unit vector away)
        ele = (alpha, beta)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(comp.max_refine)):
                continue

            # Add the new index if it maintains downward-closedness
            new_cand = (tuple(ind_new[:len(alpha)]), tuple(ind_new[len(alpha):]))
            down_closed = True
            for j in range(len(ind)):
                ind_check = ind_new.copy()
                ind_check[j] -= 1
                if ind_check[j] >= 0:
                    tup_check = (tuple(ind_check[:len(alpha)]), tuple(ind_check[len(alpha):]))
                    if tup_check not in index_set and tup_check != ele:
                        down_closed = False
                        break
            if down_closed:
                new_candidates.append(new_cand)

        # Move to the active index set
        if ele in candidate_set:
            candidate_set.remove(ele)
        index_set.append(ele)
        new_candidates = [cand for cand in new_candidates if cand not in candidate_set]
        candidate_set.extend(new_candidates)

        # Return total cost of activation
        total_cost = 0.0
        for a, b in new_candidates:
            total_cost += comp.get_cost(a, b)
        return total_cost

    # Add initialization costs
    base_alpha = (0,) * len(comp.truth_alpha)
    base_beta = (0,) * (len(comp.max_refine) - len(comp.truth_alpha))
    base_cost = activate_index(base_alpha, base_beta) + comp.get_cost(base_alpha, base_beta)
    cost_cum.append(base_cost)

    # Add cumulative training costs
    for i in range(surr.refine_level):
        err_indicator, node, alpha, beta, num_evals, cost = surr.build_metrics['train_record'][i]
        new_cost = activate_index(alpha, beta)
        cost_cum.append(float(new_cost))

    return np.cumsum(cost_cum)


def plot_surrogate():
    """Make extra plots of the surrogate for the journal paper"""
    # Error v. cost
    sf_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-03-07T01.53.07' / 'single-fidelity').glob('amisc_*'))[0]
    sf_sys = pem_v0(from_file=sf_dir / 'sys' / f'sys_final.pkl')
    mf_sys = SURR

    sf_test = sf_sys.build_metrics['test_stats']  # (Niter+1, 2, Nqoi)
    mf_test = mf_sys.build_metrics['test_stats']  # (Niter+1, 2, Nqoi)
    sf_cum = get_allocation(sf_sys)               # (Niter+1,)
    mf_cum = get_allocation(mf_sys)               # (Niter+1,)

    sf_alloc, _, _ = sf_sys.get_allocation()
    hf_alloc = sf_alloc['Thruster'][str(tuple())]  # [Neval, total cost]
    hf_model_cost = hf_alloc[1] / hf_alloc[0]

    rc = {'axes.labelsize': 23, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18}
    figsize = (6, 5)
    with plt.style.context('uqtils.default'):
        with matplotlib.rc_context(rc=rc):
            figs, axs = zip(*[plt.subplots(figsize=figsize, layout='tight') for i in range(3)])
            labels = ['discharge', 'thrust', 'uion0']
            for i in range(3):
                ax = axs[i]
                ax.plot(mf_cum / hf_model_cost, mf_test[:, 1, i], '-k', label='Multi-fidelity (MF)')
                ax.plot(sf_cum / hf_model_cost, sf_test[:, 1, i], '--k', label='Single-fidelity (SF)')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_ylim([5e-2, 2])
                ylabel = r'Relative $\mathrm{L}_2$ error' if i == 0 else ''
                uq.ax_default(ax, r'Cost (number of SF evals)', ylabel, legend=i == 2)
                if i > 0:
                    ax.axes.yaxis.set_ticklabels([])
                figs[i].savefig(f'{labels[i]}-error_v_cost.pdf', bbox_inches='tight', format='pdf')
            plt.show()

        # 1d slice 4x4 gird
        fig, ax = SURR.plot_slice(from_file=surr_dir / 'sweep_NYYA' / 'sweep_randNYYA_s9,10,12,13_q2,3,9,10.pkl')
        fig.set_size_inches(14, 14)
        fig.tight_layout()
        fig.savefig('sweep_randNYYA_s9,10,12,13_q2,3,9,10.pdf', bbox_inches='tight', format='pdf')
        plt.show()


if __name__ == '__main__':
    # run_models()
    spt100_monte_carlo(plot=False)
    # plot_surrogate()
