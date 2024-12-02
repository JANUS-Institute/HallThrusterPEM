"""Script for doing Sobol' sensitivity analysis."""
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.stats as st
import h5py
import uqtils as uq
import matplotlib.pyplot as plt
import matplotlib

from hallmd.models.pem import pem_v0
from hallmd.models.thruster import uion_reconstruct

PROJECT_ROOT = Path('../..')
TRAINING = False
SAMPLER = 'Posterior'  # Or 'Prior'
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-03-07T01.53.07' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')
COMP = 'System'
CONSTANTS = {'Va', 'r_m'}

SKIP_IDX = [i for i, v in enumerate(SURR.x_vars) if str(v) in CONSTANTS]
LCH = 0.025
IDX_MAP = {'V_cc': [i for i in SURR.graph.nodes['Cathode']['exo_in'] if i not in SKIP_IDX],
           'T': [i for i in SURR.graph.nodes['Thruster']['exo_in'] if i not in SKIP_IDX],
           'uion': [i for i in SURR.graph.nodes['Thruster']['exo_in'] if i not in SKIP_IDX],
           'jion': [i for i in SURR.graph.nodes['Plume']['exo_in'] if i not in SKIP_IDX]
           }
QOIS = ['V_cc', 'T', 'uion', 'jion']

MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('xtick.major', size=6, width=1, pad=6)
plt.rc('ytick.major', size=6, width=1, pad=6)
plt.rc('xtick.minor', size=3, width=0.8, visible=True)
plt.rc('ytick.minor', size=3, width=0.8, visible=True)


def compute_indices(Ns=1000):
    def sampler(shape, nominal, qoi: Literal['V_cc', 'T', 'uion', 'jion']):
        x = SURR.sample_inputs(shape, use_pdf=True, nominal=nominal, constants=CONSTANTS)[..., IDX_MAP.get(qoi)]

        if qoi == 'jion':
            # Handle some really bad samples of jion
            comp_in = np.empty(shape + (x.shape[-1] + 2,))
            comp_in[..., :-2] = x
            comp_in[..., -2] = 1    # r_m
            comp_in[..., -1] = 4    # I_B0
            comp_in = comp_in.reshape((-1, comp_in.shape[-1]))
            comp_out = SURR['Plume']._model(comp_in, compress=False)['y'][..., 1:]
            bad_idx = np.any(comp_out >= 200, axis=-1)
            num_reject = np.sum(bad_idx)
            while num_reject > 0:
                new_sample = SURR.sample_inputs(int(num_reject), use_pdf=True, nominal=nominal, constants=CONSTANTS)[..., IDX_MAP.get(qoi)]
                comp_in[bad_idx, :-2] = new_sample
                comp_out = SURR['Plume']._model(comp_in, compress=False)['y'][..., 1:]
                bad_idx = np.any(comp_out >= 200, axis=-1)
                num_reject = np.sum(bad_idx)
            x = comp_in[:, :-2].reshape(x.shape)

        return x

    def model(x, qoi: Literal['V_cc', 'T', 'uion', 'jion']):
        """Compute Vcc, T, uion(z=Lch), or jion(gamma=0)"""
        surr_in = np.empty(x.shape[:-1] + (len(SURR.x_vars),))
        j = 0
        for i in range(surr_in.shape[-1]):
            if i in IDX_MAP.get(qoi):
                surr_in[..., i] = x[..., j]
                j += 1
            else:
                surr_in[..., i] = SURR.x_vars[i].nominal
        qoi_ind = [qoi] if qoi in ['V_cc', 'T'] else [i for i, v in enumerate(SURR.coupling_vars) if str(v).startswith(qoi)]

        if qoi == 'jion':
            comp_in = np.empty(x.shape[:-1] + (x.shape[-1] + 2,))
            comp_in[..., :-2] = x
            comp_in[..., -2] = 1  # r_m
            comp_in[..., -1] = 4  # I_B0
            surr_out = SURR['Plume']._model(comp_in, compress=False)['y'][..., 1:]
            surr_out = surr_out[..., 0, np.newaxis]
            thresh = np.percentile(surr_out, 99)
            surr_out[surr_out > thresh] = thresh  # Fix some numerical issues with jion spikes (shit model anyways)
        else:
            surr_out = SURR.predict(surr_in, training=TRAINING, qoi_ind=qoi_ind)

        if qoi == 'uion':
            z, surr_out = uion_reconstruct(surr_out)
            surr_out = surr_out[..., np.argmin(np.abs(z-LCH)), np.newaxis]

        return surr_out

    with h5py.File(f'sobol.h5', 'a') as fd:
        Nx = 5
        pb = np.linspace(-6, -4, Nx)

        for i, qoi in enumerate(QOIS):
            xlabels = [str(v) for i, v in enumerate(SURR.x_vars) if i in IDX_MAP.get(qoi)]
            S1 = np.empty((Ns, Nx, len(xlabels)))
            ST = np.empty((Ns, Nx, len(xlabels)))

            for j in range(Nx):
                nominal = {'PB': pb[j], 'Va': 300, 'mdot_a': 5}
                f_sampler = lambda shape: sampler(shape, nominal, qoi)
                f_model = lambda x: model(x, qoi)
                s1, st = uq.sobol_sa(f_model, f_sampler, Ns, qoi_labels=[qoi], param_labels=xlabels, compute_s2=False)
                S1[:, j, :] = s1[..., 0]
                ST[:, j, :] = st[..., 0]
            fd.create_dataset(f'{qoi}/S1', data=S1)
            fd.create_dataset(f'{qoi}/ST', data=ST)
        fd.create_dataset('PB', data=10 ** pb)


def spt100_sobol(Ns=1000):
    """Plot Sobol' sensitivity indices as function of pressure."""
    file = Path('sobol.h5')
    if not file.is_file():
        compute_indices(Ns)

    z = st.norm.ppf(1 - (1 - 0.95) / 2)
    figsize = (6, 5)

    with h5py.File(file, 'r') as fd, plt.style.context('uqtils.default'):
        pb = np.array(fd['PB'])
        fig1, ax1 = plt.subplots(figsize=figsize, layout='tight')
        fig2, ax2 = plt.subplots(figsize=figsize, layout='tight')
        axs = [ax1, ax2]
        for i, qoi in enumerate(['T', 'uion']):
            ax = axs[i]
            S1 = np.array(fd[f'{qoi}/S1'])
            ST = np.array(fd[f'{qoi}/ST'])
            S1_avg = np.mean(S1, axis=0)
            S1_se = np.sqrt(np.var(S1, axis=0) / Ns)
            ST_avg = np.mean(ST, axis=0)
            ST_se = np.sqrt(np.var(ST, axis=0) / Ns)
            xlabels = [v.to_tex(units=False) for i, v in enumerate(SURR.x_vars) if i in IDX_MAP.get(qoi)]
            c = plt.get_cmap('tab10')(np.linspace(0, 1, len(xlabels)))
            for j, str_var in enumerate(xlabels):
                label = str_var if i == 1 else None
                ax.errorbar(pb, S1_avg[:, j], yerr=z * S1_se[:, j], ls='-', capsize=3, label=label, color=c[j])
                ax.errorbar(pb, ST_avg[:, j], yerr=z * ST_se[:, j], ls='--', capsize=3, color=c[j])
            ax.plot(np.nan, np.nan, ls='-', color=(0.5, 0.5, 0.5), alpha=0.5, label=f'$S_1$' if i==1 else None)
            ax.plot(np.nan, np.nan, ls='--', color=(0.5, 0.5, 0.5), alpha=0.5, label=f'$S_T$' if i==1 else None)
            ax.set_xscale('log')
            ax.set_ylim(bottom=0, top=1.05)
            uq.ax_default(ax, 'Background pressure (Torr)', "Sobol' index", legend=False)
        with matplotlib.rc_context(rc={'legend.fontsize': 14}):
            leg = axs[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.025), labelspacing=0.8)
        fig1.savefig(f'sobol-thrust.pdf', bbox_inches='tight', format='pdf')
        fig2.savefig(f'sobol-uion.pdf', bbox_inches='tight', format='pdf')
        plt.show()

        for i, qoi in enumerate(['V_cc', 'jion']):
            S1 = np.array(fd[f'{qoi}/S1'])
            ST = np.array(fd[f'{qoi}/ST'])
            S1_avg = np.mean(S1, axis=0)
            S1_se = np.sqrt(np.var(S1, axis=0) / Ns)
            ST_avg = np.mean(ST, axis=0)
            ST_se = np.sqrt(np.var(ST, axis=0) / Ns)

            fig, ax = plt.subplots(layout='tight', figsize=figsize)
            xlabels = [v.to_tex(units=False) for i, v in enumerate(SURR.x_vars) if i in IDX_MAP.get(qoi)]
            c = plt.get_cmap('tab10')(np.linspace(0, 1, len(xlabels)))
            for j, str_var in enumerate(xlabels):
                ax.errorbar(pb, S1_avg[:, j], yerr=z * S1_se[:, j], ls='-', capsize=3, label=str_var, color=c[j])
                ax.errorbar(pb, ST_avg[:, j], yerr=z * ST_se[:, j], ls='--', capsize=3, color=c[j])
            ax.plot(np.nan, np.nan, ls='-', color=(0.5, 0.5, 0.5), alpha=0.5, label=f'$S_1$')
            ax.plot(np.nan, np.nan, ls='--', color=(0.5, 0.5, 0.5), alpha=0.5, label=f'$S_T$')
            ax.set_xscale('log')
            ax.set_ylim(bottom=0, top=1.08)
            uq.ax_default(ax, 'Background pressure (Torr)', "Sobol' index", legend=qoi == 'V_cc')
            if qoi == 'jion':
                leg = ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.025), labelspacing=0.8)
            fig.savefig(f'sobol-{qoi.lower()}.pdf', bbox_inches='tight', format='pdf')
            plt.show()


if __name__ == '__main__':
    spt100_sobol(Ns=5000)
