"""Test the ion current density plume model."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from hallmd.models.plume import current_density

SHOW_PLOTS = False  # for debugging
J_MIN = 1e-6        # Minimum expected ion current density
J_MAX = 1e3         # Maximum expected ion current density
N = 100             # Number of grid points


def test_random_samples(plots=SHOW_PLOTS):
    # Test vectorized usage on random samples
    inputs_rand = {'P_b': 10 ** (np.random.rand(N) * 4 - 8), 'c0': np.random.rand(N) * 0.8 + 0.1,
                   'c1': np.random.rand(N) * 0.8 + 0.1, 'c2': np.random.rand(N) * 30 - 15,
                   'c3': np.random.rand(N) + 0.1, 'c4': 10 ** (np.random.rand(N) * 4 + 18),
                   'c5': 10 ** (np.random.rand(N) * 4 + 14), 'sigma_cex': np.random.rand(N) * 7e-20 + 51e-20,
                   'r_p': np.random.rand(N)*0.4 + 0.8, 'I_B0': np.random.rand(N) * 6 + 2}
    outputs_rand = current_density(inputs_rand)

    min_jion = np.min(outputs_rand['j_ion'])
    max_jion = np.max(outputs_rand['j_ion'])

    print(f'Minimum ion current density: {min_jion:.2e} A/m^2')
    print(f'Maximum ion current density: {max_jion:.2e} A/m^2')

    assert min_jion >= J_MIN
    assert max_jion <= J_MAX

    if plots:
        # Plot bounds of random samples
        jion = outputs_rand['j_ion']
        alpha_deg = np.linspace(0, np.pi / 2, jion.shape[-1]) * (180 / np.pi)
        lb = np.percentile(jion, 5, axis=0)
        mid = np.percentile(jion, 50, axis=0)
        ub = np.percentile(jion, 95, axis=0)

        fig, ax = plt.subplots()
        ax.plot(alpha_deg, mid, '-k')
        ax.fill_between(alpha_deg, lb, ub, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
        ax.set_yscale('log')
        ax.set_xlabel('Angle from centerline (deg)')
        ax.set_ylabel('Ion current density ($A/m^2$)')

        plt.show()


def test_pressure_sweep(plots=SHOW_PLOTS):
    # Test single 1d sweep
    pressure_sweep = 10 ** (np.linspace(-6, -4, N))
    inputs_sweep = {'P_b': pressure_sweep, 'c0': 0.1, 'c1': 0.7, 'c2': -8.0, 'c3': 0.2,
                    'c4': 1e20, 'c5': 1e16, 'sigma_cex': 55e-20, 'r_p': 1, 'I_B0': 3}
    outputs_sweep = current_density(inputs_sweep)

    min_jion = np.min(outputs_sweep['j_ion'])
    max_jion = np.max(outputs_sweep['j_ion'])

    print(f'Minimum ion current density: {min_jion:.2e} A/m^2')
    print(f'Maximum ion current density: {max_jion:.2e} A/m^2')

    assert min_jion >= J_MIN
    assert max_jion <= J_MAX

    # Make sure total current is invariant
    R = 1  # m
    theta = np.linspace(0, np.pi / 2, outputs_sweep['j_ion'].shape[-1])
    current = np.empty(outputs_sweep['j_ion'].shape[0])
    for i in range(outputs_sweep['j_ion'].shape[0]):
        current[i] = 2 * np.pi * R**2 * simpson(outputs_sweep['j_ion'][i, :] * np.sin(theta), x=theta)

    err = np.sqrt(np.sum((current - np.mean(current)) ** 2) / np.sum(current ** 2))
    assert err < 1e-4

    if plots:
        # Plot trend of divergence angle with pressure
        fig, ax = plt.subplots()
        div_angle = outputs_sweep['div_angle'] * (180 / np.pi)  # deg
        ax.plot(pressure_sweep, div_angle, '-k')
        ax.set_xscale('log')
        ax.set_xlabel('Background pressure (Torr)')
        ax.set_ylabel('Divergence angle (deg)')

        # Plot trend of jion with pressure
        fig, ax = plt.subplots()
        jion = outputs_sweep['j_ion']
        alpha_deg = np.linspace(0, np.pi / 2, jion.shape[-1]) * (180 / np.pi)
        skip = 10
        c = plt.get_cmap('viridis')(np.linspace(0, 1, jion.shape[0]))
        for i in range(0, jion.shape[0], skip):
            ax.plot(np.concatenate((-np.flip(alpha_deg)[:-1], alpha_deg)),
                    np.concatenate((np.flip(jion[i, :-1], axis=-1), jion[i, :]), axis=-1),
                    label=f'P_b = {pressure_sweep[i]:.2e}', color=c[i])
        ax.set_yscale('log')
        ax.set_xlabel('Angle from centerline (deg)')
        ax.set_ylabel('Ion current density ($A/m^2$)')
        ax.legend()

        plt.show()
