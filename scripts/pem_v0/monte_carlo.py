"""Script for surrogate-enabled Monte Carlo forward UQ analysis."""
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py
import uqtils as uq
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.stats as st
from itertools import cycle

from hallmd.data.loader import spt100_data
from hallmd.models.plume import jion_reconstruct
from hallmd.models.thruster import uion_reconstruct
from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

PROJECT_ROOT = Path('../..')
TRAINING = False
SAMPLER = 'Posterior'  # Or 'Prior'
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-02-14T06.54.13' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')
DATA = spt100_data()
COMP = 'System'
THETA_VARS = [v for v in SURR[COMP].x_vars if v.param_type == 'calibration']
QOI_MAP = {'Cathode': ['V_cc'], 'Thruster': ['T', 'uion'], 'Plume': ['jion'], 'System': ['V_cc', 'T', 'uion', 'jion']}
QOIS = QOI_MAP.get(COMP)
CONSTANTS = {"calibration"} if SAMPLER == 'Posterior' else set()
CONSTANTS.update({"delta_z", "vAN1", "r_m"})

with open(model_config_dir() / 'plume_svd.pkl', 'rb') as fd:
    PLUME_SVD = pickle.load(fd)
with open(model_config_dir() / 'thruster_svd.pkl', 'rb') as fd:
    THRUSTER_SVD = pickle.load(fd)

# Load MCMC results
with h5py.File(f'dram-{COMP.lower()}-{"train" if TRAINING else "test"}.h5', 'r') as fd:
    SAMPLES = fd['mcmc/chain']
    niter, nwalk, ndim = SAMPLES.shape
    SAMPLES = SAMPLES[int(0.1 * niter):, ...].reshape((-1, ndim))


def mcmc_sampler(shape):
    if not isinstance(shape, tuple):
        shape = (shape,)
    i = np.random.randint(0, SAMPLES.shape[0], shape)
    return SAMPLES[i, :]


def run_models(Ns=1000):
    """Get full-model and surrogate predictions and save to file"""
    with h5py.File(f'monte-carlo.h5', 'a') as fd:
        # Cathode coupling voltage
        data = DATA['V_cc'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        if SAMPLER == 'Posterior':
            theta = mcmc_sampler(sample_shape)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=QOI_MAP.get('Cathode'), training=TRAINING), axis=-1)
        fd.create_dataset('vcc/xs', data=xs)
        fd.create_dataset('vcc/ys', data=ys)

        # Thrust (Diamant)
        data = DATA['T'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        if SAMPLER == 'Posterior':
            theta = mcmc_sampler(sample_shape)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=['T'], training=TRAINING), axis=-1)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = np.squeeze(SURR.predict(xmodel, qoi_ind=['T'], use_model='best'), axis=-1)
        fd.create_dataset('thrust/xs', data=xs)
        fd.create_dataset('thrust/ys', data=ys)
        fd.create_dataset('thrust/xmodel', data=xmodel)
        fd.create_dataset('thrust/ymodel', data=ymodel)
        idx = np.argsort(pb)[-5]
        nominal = {'PB': pb[idx], 'Va': data['x'][idx, 1], 'mdot_a': data['x'][idx, 2]}
        xs = np.zeros((100, Ns, len(SURR.x_vars)))
        for i in range(100):
            theta = mcmc_sampler(1)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
            xs[i, ...] = SURR.sample_inputs(Ns, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=['T'], training=TRAINING), axis=-1)
        fd.create_dataset('thrust-uq/xs', data=xs)
        fd.create_dataset('thrust-uq/ys', data=ys)

        # Thrust (Sankovic)
        data = DATA['T'][1]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': data['x'][:, 0], 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        if SAMPLER == 'Posterior':
            theta = mcmc_sampler(sample_shape)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = np.squeeze(SURR.predict(xs, qoi_ind=['T'], training=TRAINING), axis=-1)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        ymodel = np.squeeze(SURR.predict(xmodel, qoi_ind=['T'], use_model='best'), axis=-1)
        fd.create_dataset('thrust-test/xs', data=xs)
        fd.create_dataset('thrust-test/ys', data=ys)
        fd.create_dataset('thrust-test/xmodel', data=xmodel)
        fd.create_dataset('thrust-test/ymodel', data=ymodel)

        # Ion velocity
        data = DATA['uion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        if SAMPLER == 'Posterior':
            theta = mcmc_sampler(sample_shape)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        qois = [var for var in SURR.coupling_vars if str(var).startswith('uion')]
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        vcc = SURR.predict(xmodel, qoi_ind=QOI_MAP.get('Cathode'), use_model='best')
        thruster_in = np.concatenate((xmodel[..., SURR.graph.nodes['Thruster']['exo_in']], vcc), axis=-1)
        ymodel = SURR['Thruster']._model(thruster_in, alpha=(2, 2), compress=False, svd_data=THRUSTER_SVD)['y']
        fd.create_dataset('uion/xs', data=xs)
        fd.create_dataset('uion/ys', data=ys)
        fd.create_dataset('uion/xmodel', data=xmodel)
        fd.create_dataset('uion/ymodel', data=ymodel[..., 6:])

        # Ion current density
        data = DATA['jion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        sample_shape = (Ns, Nx)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        if SAMPLER == 'Posterior':
            theta = mcmc_sampler(sample_shape)
            nominal.update({str(v): theta[..., i] for i, v in enumerate(THETA_VARS)})
        qois = [var for var in SURR.coupling_vars if str(var).startswith('jion')]
        xs = SURR.sample_inputs(sample_shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)
        ys = SURR.predict(xs, qoi_ind=qois, training=TRAINING)
        nominal = {'PB': pb, 'Va': data['x'][:, 1], 'mdot_a': data['x'][:, 2]}
        xmodel = SURR.sample_inputs(Nx, use_pdf=False, nominal=nominal,
                                    constants=CONSTANTS.union({"design", "other", "operating"}))
        I_B0 = SURR.predict(xmodel, qoi_ind=['I_B0'], use_model='best')
        plume_in = np.concatenate((xmodel[..., SURR.graph.nodes['Plume']['exo_in']], I_B0), axis=-1)
        ymodel = SURR['Plume']._model(plume_in, compress=False, svd_data=PLUME_SVD)['y']
        alpha_g = np.linspace(0, np.pi / 2, 100)
        jion_g = ymodel[..., 1:]
        alpha_g2 = np.concatenate((-np.flip(alpha_g)[:-1], alpha_g))  # (2M-1,)
        jion_g2 = np.concatenate((np.flip(jion_g, axis=-1)[..., :-1], jion_g), axis=-1)  # (..., 2M-1)
        f = interp1d(alpha_g2, jion_g2, axis=-1)
        jion_interp = f(data['loc'][:, 1])  # (..., Nx)
        fd.create_dataset('jion/xs', data=xs)
        fd.create_dataset('jion/ys', data=ys)
        fd.create_dataset('jion/xmodel', data=xmodel)
        fd.create_dataset('jion/ymodel', data=jion_interp)


def spt100_monte_carlo(Ns=1000):
    """Plot `[V_cc, T, uion, jion]` against SPT-100 experimental data with UQ bounds."""
    file = Path('monte-carlo.h5')
    if not file.is_file():
        run_models(Ns)

    with h5py.File(file, 'r') as fd:
        # Cathode coupling voltage
        data = DATA['V_cc'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        ys = fd['vcc/ys']
        fig, ax = plt.subplots()
        yerr = 2 * np.sqrt(data['var_y'])
        ax.errorbar(10 ** pb, data['y'], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none',
                    label='Experiment', markersize=4)
        p5 = np.percentile(ys, 5, axis=0)
        med = np.percentile(ys, 50, axis=0)
        p95 = np.percentile(ys, 95, axis=0)
        ax.fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        ax.plot(10 ** pb[idx], med[idx], '-k', label='Model')
        ax.grid()
        uq.ax_default(ax, 'Background pressure (Torr)', 'Cathode coupling voltage (V)', legend=False)
        leg = ax.legend(loc='upper left', fancybox=True, facecolor='white', framealpha=1)
        frame = leg.get_frame()
        frame.set_edgecolor('k')
        ax.set_xscale('log')
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        fig.savefig('mc-vcc.png', dpi=300, format='png')
        plt.show()

        # Thrust (Diamant)
        data = DATA['T'][0]
        pb = data['x'][:, 0]
        idx = np.argsort(pb)
        ys = fd['thrust/ys']
        ymodel = np.array(fd['thrust/ymodel'])
        fig, ax = plt.subplots(1, 3)
        yerr = 2 * np.sqrt(data['var_y'])
        ax[0].errorbar(10 ** pb, data['y'] * 1000, yerr=yerr*1000, fmt='or', capsize=3, markerfacecolor='none',
                    label='Experiment', markersize=4)
        p5 = np.percentile(ys, 5, axis=0) * 1000
        med = np.percentile(ys, 50, axis=0) * 1000
        p95 = np.percentile(ys, 95, axis=0) * 1000
        ax[0].fill_between(10 ** pb[idx], p5[idx], p95[idx], alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        ax[0].plot(10 ** pb[idx], med[idx], '-k', label='Surrogate')
        ax[0].plot(10 ** pb[idx], ymodel[idx] * 1000, '--k', label='Model')
        ax[0].grid()
        xline = 10 ** (np.ones(5)*pb[idx][-5])
        yline = np.linspace(77.5, 85, 5)
        ax[0].plot(xline, yline, '-.b', linewidth=1.2, alpha=0.6)
        xy = (xline[0]-0.1e-5, yline[0])
        ax[0].annotate(f'A', xy, xytext=(xy[0] + 1e-5, xy[1]-0.1), weight='bold',
                       arrowprops={'arrowstyle': '<|-', 'linewidth': 1.2, 'alpha': 0.6, 'color': 'b'})
        xy = (xline[0]-0.1e-5, yline[-1])
        ax[0].annotate(f'A', xy, xytext=(xy[0] + 1e-5, xy[1]-0.1), weight='bold',
                       arrowprops={'arrowstyle': '<|-', 'linewidth': 1.2, 'alpha': 0.6, 'color': 'b'})
        uq.ax_default(ax[0], 'Background pressure (Torr)', 'Thrust (mN)', legend=False)
        leg = ax[0].legend(loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        frame = leg.get_frame()
        frame.set_edgecolor('k')
        ax[0].set_xscale('log')
        ax[1].set_title(r'Section A-A at $P_B=3.81\text{E-5}$ Torr')
        ax[1].set_xlim((yline[0], yline[-1]))
        ax[1].hist(ys[:, idx[-5]]*1000, bins=15, density=True, facecolor=[0.8, 0.8, 0.8], alpha=0.4, edgecolor='k', lw=1.4, label='Surrogate')
        ax[1].axvline(ymodel[idx[-5]]*1000, c='k', ls='--', label='Model')
        exp_y = data['y'][idx[-5]] * 1000
        std = yerr[idx[-5]] * 1000 / 2
        x = np.linspace(exp_y - 3.5*std, exp_y + 3.5*std, 100)
        y = np.squeeze(uq.normal_pdf(x[..., np.newaxis], exp_y, std**2))
        ax[1].plot(x, y, '-r', label='Experiment')
        ax[1].fill_between(x, y1=y, y2=0, lw=0, alpha=0.1, facecolor='r')
        uq.ax_default(ax[1], 'Thrust (mN)', 'PDF', legend=True)
        ys = np.array(fd['thrust-uq/ys']) * 1000  # (100, 1000), CDF plots at PB=3.81E-5 Torr
        ys = np.sort(ys, axis=-1)
        cdfs = np.arange(1, ys.shape[-1] + 1) / ys.shape[-1]
        cycol = cycle('bgrcmk')
        for i in range(0, ys.shape[0], 3):
            ax[2].plot(ys[i, :], cdfs, ls='-', c=next(cycol), alpha=0.2)
        ax[2].set_title(r'Uncertainty at $P_B=3.81\text{E-5}$ Torr')
        xy = (79.58, 0.2)
        ax[2].annotate('Epistemic uncertainty', xy, xytext=(xy[0]+0.27, xy[1]),
                       arrowprops={'arrowstyle': '<|-|>', 'linewidth': 1.4, 'color': 'k'})
        uq.ax_default(ax[2], 'Thrust (mN)', 'Aleatoric CDF', legend=False)
        fig.set_size_inches(16, 5)
        fig.tight_layout()
        fig.savefig('mc-thrust.png', dpi=300, format='png')
        plt.show()

        # Thrust (Sankovic 1993)
        data = DATA['T'][1]
        ys = np.array(fd['thrust-test/ys'])
        ymodel = np.array(fd['thrust-test/ymodel'])
        xerr = 2 * np.sqrt(data['var_y']) * 1000
        med = np.percentile(ys, 50, axis=0) * 1000
        yerr = np.vstack((med - np.percentile(ys, 5, axis=0)*1000, np.percentile(ys, 95, axis=0)*1000 - med))
        fig, ax = plt.subplots()
        ax.errorbar(data['y'] * 1000, med, yerr=yerr, xerr=xerr, fmt='ok', capsize=2, markerfacecolor='none',
                    label='Surrogate', markersize=3, alpha=0.5)
        ax.errorbar(data['y'] * 1000, ymodel*1000, xerr=xerr, fmt='or', capsize=2,
                    label='Model', markersize=3, alpha=0.5)
        ax.plot(np.sort(data['y'] * 1000), np.sort(data['y'] * 1000), '-k', alpha=0.6, linewidth=1.2)
        ax.grid()
        uq.ax_default(ax, 'Experimental thrust (mN)', 'Predicted thrust (mN)', legend=True)
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        fig.savefig('mc-thrust-test.png', dpi=300, format='png')
        plt.show()

        # Ion velocity
        data = DATA['uion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        ys = np.array(fd['uion/ys'])
        zg, ys_g = uion_reconstruct(ys)
        ymodel = np.array(fd['uion/ymodel'])
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b']
        for i in range(Nx):
            yerr = 2 * np.sqrt(data['var_y'][i, :])
            ax.errorbar(data['loc']*1000, data['y'][i, :], yerr=yerr, fmt='o', color=colors[i], capsize=2,
                        markerfacecolor='none', label=f'{10**pb[i]:.2E} Torr', markersize=2)
            p5 = np.percentile(ys_g[:, i, :], 5, axis=0)
            med = np.percentile(ys_g[:, i, :], 50, axis=0)
            p95 = np.percentile(ys_g[:, i, :], 95, axis=0)
            ax.fill_between(zg*1000, p5, p95, alpha=0.2, edgecolor=(0.4, 0.4, 0.4), facecolor=colors[i])
            ax.plot(zg*1000, med, ls='-', color=colors[i], alpha=0.5)
            ax.plot(zg*1000, ymodel[i, :], ls='--', color=colors[i], alpha=0.5)
        ax.plot(np.nan, np.nan, ls='-', color=[0.6, 0.6, 0.6], label='Surrogate')
        ax.plot(np.nan, np.nan, ls='--', color=[0.6, 0.6, 0.6], label='Model')
        ax.errorbar(np.nan, np.nan, yerr=0, fmt='o', color=[0.6, 0.6, 0.6], capsize=2, markerfacecolor='none', markersize=2, label='Experiment')
        ax.grid()
        uq.ax_default(ax, 'Axial distance from anode (mm)', 'Axial ion velocity (m/s)', legend=False)
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 3, 4, 0, 1, 5]
        leg = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                        fancybox=True, facecolor='white', framealpha=1)
        frame = leg.get_frame()
        frame.set_edgecolor('k')
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        fig.savefig('mc-uion.png', dpi=300, format='png')
        plt.show()

        # fig, ax = plt.subplots(1, Nx, sharey='row')
        # for i in range(Nx):
        #     yerr = 2 * np.sqrt(data['var_y'][i, :])
        #     ax[i].errorbar(data['loc']*1000, data['y'][i, :], yerr=yerr, fmt='or', capsize=3,
        #                    markerfacecolor='none', label='Experiment', markersize=4)
        #     ylabel = 'Axial ion velocity (m/s)' if i == 0 else ''
        #     legend = i == Nx-1
        #     p5 = np.percentile(ys_g[:, i, :], 5, axis=0)
        #     med = np.percentile(ys_g[:, i, :], 50, axis=0)
        #     p95 = np.percentile(ys_g[:, i, :], 95, axis=0)
        #     ax[i].fill_between(zg*1000, p5, p95, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        #     ax[i].plot(zg*1000, med, '-k', label='Surrogate')
        #     ax[i].plot(zg*1000, ymodel[i, :], '--k', label='Model')
        #     uq.ax_default(ax[i], 'Axial distance from anode (mm)', ylabel, legend=legend)
        #     ax[i].set_title(f'{10**pb[i]:.2E} Torr')
        # fig.set_size_inches(Nx*3, 3)
        # fig.tight_layout()
        # plt.show()

        # Ion current density
        data = DATA['jion'][0]
        pb = data['x'][:, 0]
        Nx = data['x'].shape[0]
        ys = np.array(fd['jion/ys'])
        ymodel = np.array(fd['jion/ymodel'])
        alpha_g = data['loc'][:, 1]
        alpha_i, jion_i = jion_reconstruct(ys, alpha=alpha_g)
        alpha_i = alpha_i*180/np.pi

        fig, ax = plt.subplots()
        yerr = 2 * np.sqrt(data['var_y'][0, :])
        ax.errorbar(alpha_i, data['y'][0, :], yerr=yerr, fmt='or', capsize=3, markerfacecolor='none', markersize=4, label='Experiment')
        p5 = np.percentile(jion_i[:, 0, :], 5, axis=0)
        med = np.percentile(jion_i[:, 0, :], 50, axis=0)
        p95 = np.percentile(jion_i[:, 0, :], 95, axis=0)
        ax.fill_between(alpha_i, p5, p95, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        ax.plot(alpha_i, med, '-k', label='Surrogate')
        ax.plot(alpha_i, ymodel[0, :], '--k', label='Model')
        ax.set_yscale('log')
        ax.grid()
        ax.set_title(f'$P_B$ = {10 ** pb[0]:.2E} Torr')
        uq.ax_default(ax, 'Angle from thruster centerline (deg)', r'Ion current density (A/$\text{m}^2$)', legend=False)
        leg = ax.legend(loc='upper right', fancybox=True, facecolor='white', framealpha=1)
        frame = leg.get_frame()
        frame.set_edgecolor('k')
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        fig.savefig('mc-jion-exp.png', dpi=300, format='png')
        plt.show()

        fig, ax = plt.subplots(1, 2, sharey='row')
        colors = plt.get_cmap('jet')(np.linspace(0, 1, Nx))
        for i in range(Nx):
            ax[0].plot(alpha_i, data['y'][i, :], '-o', color=colors[i], alpha=0.5, ms=3)
            ax[1].plot(alpha_i, ymodel[i, :], '--', color=colors[i], alpha=0.5)
            ax[1].plot(np.nan, np.nan, '-', color=colors[i], label=f'{10 ** pb[i]:.2E} Torr')
        ax[0].set_yscale('log')
        ax[0].grid()
        ax[0].set_title('Experiment')
        uq.ax_default(ax[0], 'Angle from thruster centerline (deg)', r'Ion current density (A/$\text{m}^2$)', legend=False)
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].set_title('Model')
        uq.ax_default(ax[1], 'Angle from thruster centerline (deg)', '', legend=True)
        fig.set_size_inches(11, 6)
        fig.tight_layout()
        fig.savefig('mc-jion-model.png', dpi=300, format='png')
        plt.show()

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
    spt100_monte_carlo()
