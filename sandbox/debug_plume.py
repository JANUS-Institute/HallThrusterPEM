import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

sys.path.append('..')

from utils import UniformRV, LogUniformRV, ax_default
from models.plume import plume_pem, jion_reconstruct
from surrogates.sparse_grids import TensorProductInterpolator
from surrogates.system import SystemSurrogate

exo_vars = [UniformRV(-8, -3, 'PB'), UniformRV(200, 400, 'Va'), UniformRV(3e-6, 7e-6, 'mdot_a'),
                UniformRV(1, 5, 'T_ec'), UniformRV(0, 60, 'V_vac'), UniformRV(-2e-5, 10e-5, 'P*'),
                UniformRV(0, 15e-5, 'PT'), UniformRV(100, 500, 'u_n'), UniformRV(0.1, 0.3, 'c_w'),
                UniformRV(0.001, 0.02, 'l_t'), LogUniformRV(-3, -1, 'vAN1'), UniformRV(2, 100, 'vAN2'),
                UniformRV(0.07, 0.09, 'l_c'), UniformRV(800, 1200, 'Ti'), UniformRV(280, 320, 'Tn'),
                UniformRV(280, 320, 'Tb'), UniformRV(0, 1, 'c0'), UniformRV(0.1, 0.9, 'c1'), UniformRV(-15, 15, 'c2'),
                UniformRV(0, np.pi/2, 'c3'), UniformRV(18, 22, 'c4'), UniformRV(14, 18, 'c5'),
                UniformRV(51, 58, 'sigma_cex'), UniformRV(0.5, 1.5, 'r_m')]
coupling_vars = [UniformRV(0, 60, 'V_cc'), UniformRV(0, 10, 'I_B0'), UniformRV(0, 0.2, 'T'),
                 UniformRV(0, 1, 'eta_v'), UniformRV(0, 1, 'eta_c'), UniformRV(0, 1, 'eta_m'),
                 UniformRV(14000, 22000, 'ui_avg')]
plume_exo = [0, 16, 17, 18, 19, 20, 21, 22, 23]

vars = [var for i, var in enumerate(exo_vars) if i in plume_exo]
vars.append(coupling_vars[1])


def sample(shape):
    x = np.zeros(shape + (len(vars),))
    for i in range(len(vars)):
        x[..., i] = vars[i].sample(shape)
    return x


def test_interp():
    beta = [1]*len(vars)
    interp = TensorProductInterpolator(beta, vars, model=plume_pem)
    interp.set_yi()

    N = 10000
    xt = sample((N,))
    yt = plume_pem(xt)
    yinterp = interp(xt)
    pct_error = 100*(np.abs(yinterp - yt)) / yt

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(pct_error[:, 0], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax[1].hist(pct_error[:, 1], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax[2].hist(pct_error[:, 2], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax_default(ax[0], 'Divergence angle (rad)', '', legend=False)
    ax_default(ax[1], 'First singular value', '', legend=False)
    ax_default(ax[2], 'Second singular value', '', legend=False)
    fig.set_size_inches(9, 3)
    fig.tight_layout()
    plt.show()

    alpha_g, jion_g = jion_reconstruct(yt[:, 2:])
    lb = np.percentile(jion_g, 5, axis=0)
    med = np.percentile(jion_g, 50, axis=0)
    ub = np.percentile(jion_g, 95, axis=0)
    fig, ax = plt.subplots()
    ax.plot(alpha_g, med, '-k')
    ax.fill_between(alpha_g, lb, ub, alpha=0.5, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    ax.set_yscale('log')
    ax_default(ax, 'Angle from centerline (rad)', 'Ion current density (A/m^2)', legend=False)
    plt.show()


def test_refine():
    coupling_vars = [UniformRV(0, np.pi / 2, 'theta_d')]  # Div angle
    with open(Path('../models/data') / 'plume_svd.pkl', 'rb') as fd:
        d = pickle.load(fd)
        r1 = d['vtr'].shape[0]
        coupling_vars.extend([UniformRV(-10, 10, f'jion{i}') for i in range(r1)])
    plume = {'name': 'Plume', 'model': plume_pem, 'truth_alpha': (), 'exo_in': list(np.arange(0, len(vars))),
             'max_alpha': (), 'local_in': {}, 'global_out': list(np.arange(0, len(coupling_vars))),
             'type': 'lagrange', 'max_beta': (3,) * len(vars)}
    sys = SystemSurrogate([plume], vars, coupling_vars, root_dir='build', est_bds=10000)
    sys.build_system(qoi_ind=[0, 1, 2], N_refine=200, max_iter=50, max_runtime=20*60, max_tol=1e-5)
    # sys = SystemSurrogate.load_from_file('build/sys/sys_final.pkl')

    p = np.linspace(-8, -3, 100).reshape((100, 1))
    c0 = np.ones((100, 1)) * 0.5
    c1 = np.ones((100, 1)) * 0.3
    c2 = np.ones((100, 1)) * 9.73
    c3 = np.ones((100, 1)) * 0.261
    c4 = np.ones((100, 1)) * 19.5
    c5 = np.ones((100, 1)) * 16
    sigma_cex = np.ones((100, 1)) * 55
    r_m = np.ones((100, 1)) * 1
    I_B0 = np.ones((100, 1)) * 3.6
    xp = np.concatenate((p, c0, c1, c2, c3, c4, c5, sigma_cex, r_m, I_B0), axis=-1)
    y_truth = plume_pem(xp)
    y_surr = sys(xp)

    def plot_p(index):
        fig, ax = plt.subplots()
        ax.plot(np.squeeze(p), y_truth[:, index], '-k', label='Model')
        ax.plot(np.squeeze(p), y_surr[:, index], '--r', label='Surrogate')
        ax_default(ax, 'Pressure magnitude (Torr)', f'QoI {index}', legend=True)
        plt.show()

    for i in range(3):
        plot_p(i)

    N = 10000
    xt = sample((N,))
    yt = plume_pem(xt)
    ysurr = sys(xt)
    pct_error = 100 * (np.abs(ysurr - yt)) / np.abs(yt)

    fig, ax = plt.subplots(1, 3)
    ax[0].hist(pct_error[:, 0], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax[1].hist(pct_error[:, 1], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax[2].hist(pct_error[:, 2], color='red', edgecolor='black', linewidth=1.2, density=True)
    ax_default(ax[0], 'Divergence angle (rad)', '', legend=False)
    ax_default(ax[1], 'First singular value', '', legend=False)
    ax_default(ax[2], 'Second singular value', '', legend=False)
    fig.set_size_inches(9, 3)
    fig.tight_layout()
    plt.show()

    print('5th percentile error:')
    print(np.percentile(pct_error, 5, axis=0))
    print('Mean percent error:')
    print(np.mean(pct_error, axis=0))
    print('Median percent error:')
    print(np.percentile(pct_error, 50, axis=0))
    print('95th percentile error:')
    print(np.percentile(pct_error, 95, axis=0))


if __name__ == '__main__':
    test_interp()
    # test_refine()
