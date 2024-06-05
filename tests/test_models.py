"""Testing for wrapper models."""
from amisc.utils import load_variables
import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default

from hallmd.models.examples import chatgpt_system
from hallmd.models.thruster import hallthruster_jl_wrapper
from hallmd.models.pem import pem_v0
from hallmd.models.cc import cc_feedforward
from hallmd.models.plume import plume_feedforward
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def test_example():
    surr = chatgpt_system()
    surr.fit(max_iter=10)
    surr.plot_slice(slice_idx=[0, 1, 2])


def test_plume(plots=False):
    variables = load_variables(["PB", "c0", "c1", "c2", "c3", "c4", "c5", "sigma_cex", "r_m", "I_B0"],
                               CONFIG_DIR / 'variables_v0.json')
    N = 100
    x = np.empty((N, len(variables)))
    for i, var in enumerate(variables):
        x[:, i] = var.sample(N)
    res = plume_feedforward(x)
    jion = res['y'][:, 1:]
    lb = np.percentile(jion, 5, axis=0)
    mid = np.percentile(jion, 50, axis=0)
    ub = np.percentile(jion, 95, axis=0)
    alpha = np.linspace(0, 90, jion.shape[1])

    if plots:
        fig, ax = plt.subplots()
        ax.plot(alpha, mid, '-k')
        ax.fill_between(alpha, lb, ub, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        ax.set_yscale('log')
        ax_default(ax, 'Angle from centerline (deg)', 'Ion current density ($A/m^2$)', legend=False)
        plt.show()


def test_cc(plots=False):
    variables = load_variables(["PB", "Va", "T_ec", "V_vac", "P*", "PT"], CONFIG_DIR / 'variables_v0.json')
    N = 100
    x = np.empty((N, len(variables)))
    for i, var in enumerate(variables):
        x[:, i] = var.sample(N)
    res = cc_feedforward(x)

    if plots:
        fig, ax = plt.subplots()
        ax.hist(np.squeeze(res['y']), linewidth=1.2, edgecolor='black', color='r', density=True, bins=20)
        ax_default(ax, 'Cathode coupling voltage (V)', 'Density', legend=False)
        plt.show()


def test_hallthruster_jl(plots=False):
    variables = load_variables(["PB", "Va", "mdot_a", "T_ec", "u_n", "l_t", "vAN1", "vAN2", "delta_z",
                                "z0", "p0", "V_cc", "f_n"], CONFIG_DIR / 'variables_v0.json')
    x = np.empty((2, len(variables)))
    for i, var in enumerate(variables):
        x[:, i] = var.sample_domain(2)
    alpha = (0, 0)
    res = hallthruster_jl_wrapper(x, alpha, n_jobs=2)

    if plots:
        fig, ax = plt.subplots()
        uion = res['y'][:, 7:]
        zg = np.linspace(0, 0.08, uion.shape[1])
        for i in range(x.shape[0]):
            ax.plot(zg, uion[i, :], label=f"Sample {i}")
        ax_default(ax, 'Axial coordinate (m)', 'Ion velocity (m/s)', legend=True)
        plt.show()


def test_pem_v0():
    surr = pem_v0(init=False)
    xt = surr.sample_inputs(10)
    assert xt.shape[-1] == len(surr.exo_vars)
