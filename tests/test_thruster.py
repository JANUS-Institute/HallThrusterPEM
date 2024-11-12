"""Test the thruster model."""
import numpy as np
import matplotlib.pyplot as plt

# from hallmd.models.thruster import hallthruster_jl_wrapper
# from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def test_hallthruster_jl(plots=False):
    """This test must coincide with the `required_inputs` of the config file passed to `hallthruster_jl_wrapper`"""
    variables = load_variables(["PB", "Va", "mdot_a", "T_ec", "u_n", "l_t", "vAN1", "vAN2", "delta_z",
                                "z0", "p0", "V_cc"], CONFIG_DIR / 'variables_v0.json')
    x = np.empty((2, len(variables)))
    for i, var in enumerate(variables):
        x[:, i] = var.sample_domain(2)
    alpha = (0, 0)
    res = hallthruster_jl_wrapper(x, alpha, n_jobs=2, config=CONFIG_DIR / 'hallthruster_jl.json')

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
