"""Test the cathode coupling model."""

import matplotlib.pyplot as plt
import numpy as np

from hallmd.models.cathode import cathode_coupling


def test_cathode_coupling(tmp_path, plots=False):
    """Test prediction of cathode coupling voltage."""
    VCC_LB = 0
    VCC_UB = 100

    # Test scalar usage
    inputs = {'P_b': 10e-6, 'V_a': 300.0, 'T_e': 3.0, 'V_vac': 30.0, 'Pstar': 20e-6, 'P_T': 50e-6}
    _ = cathode_coupling(inputs, output_path=tmp_path)  # noqa: F841

    # Test vectorized usage
    N = 100
    inputs_rand = {
        'P_b': 10 ** (np.random.rand(N) * 4 - 8),
        'V_a': np.random.rand(N) * 200 + 200,
        'T_e': np.random.rand(N) * 4 + 1,
        'V_vac': np.random.rand(N) * 60,
        'Pstar': np.random.rand(N) * 90e-6 + 10e-6,
        'P_T': np.random.rand(N) * 90e-6 + 10e-6,
    }
    outputs_rand = cathode_coupling(inputs_rand, output_path=tmp_path)

    assert np.all(outputs_rand['V_cc'] >= VCC_LB) and np.all(outputs_rand['V_cc'] <= VCC_UB)

    # Test single 1d sweep
    inputs_sweep = {
        'P_b': 10 ** (np.linspace(-6, -4, N)),
        'V_a': 300,
        'T_e': 1.33,
        'V_vac': 31.6,
        'Pstar': 24.6e-6,
        'P_T': 10.2e-6,
    }
    outputs_sweep = cathode_coupling(inputs_sweep, output_path=tmp_path)

    assert np.all(outputs_sweep['V_cc'] >= VCC_LB) and np.all(outputs_sweep['V_cc'] <= VCC_UB)

    if plots:
        fig, ax = plt.subplots()
        ax.hist(outputs_rand['V_cc'], linewidth=1.2, edgecolor='black', color='r', density=True, bins=20)
        ax.set_xlabel('Cathode coupling voltage (V)')

        fig, ax = plt.subplots()
        ax.plot(inputs_sweep['P_b'], outputs_sweep['V_cc'], '-k')
        ax.set_xscale('log')
        ax.set_xlabel('Background pressure (Torr)')
        ax.set_ylabel('Cathode coupling voltage (V)')

        plt.show()
