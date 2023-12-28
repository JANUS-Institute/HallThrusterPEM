"""Module for simple algebraic models for testing purposes."""
import numpy as np

from amisc.rv import UniformRV
from amisc.system import SystemSurrogate, ComponentSpec


def ishigami(x, a=7.0, b=0.1):
    """For testing Sobol indices: https://doi.org/10.1109/ISUMA.1990.151285"""
    return {'y': np.sin(x[..., 0:1]) + a*np.sin(x[..., 1:2])**2 + b*(x[..., 2:3]**4)*np.sin(x[..., 0:1])}


def fake_pem():
    """Simple algebraic models for mocking a Hall thruster PEM."""
    def cathode(x):
        PB = x[..., 0:1]
        V_vac = 30
        Te = 2
        PT = 4e-06*133.33
        Pstar = 3e-05*133.33
        Vcc = V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB
        return dict(y=Vcc)

    def thruster(x):
        PB = x[..., 0:1]        # Pa
        Vd = x[..., 1:2]        # V
        Vcc = x[..., 2:3]       # V
        n_n = x[..., 3:4]       # 1/m^3
        gamma_n = x[..., 4:5]   # 1/m^3-s
        u_n = gamma_n / n_n     # m/s
        R = -10 * np.log10(PB/1e-4) + 90    # Ohms
        IB0 = (Vd - Vcc) / R                # A
        T1 = 0.083 - 0.003563*np.exp(-768 * PB)
        T2 = 0.0005 * (Vd - Vcc - 270)
        T3 = 0.0005 * u_n  # N
        T = T1 + T2 + T3
        y = np.concatenate((IB0, T), axis=-1)
        return dict(y=y)

    def plume(x):
        PB = x[..., 0:1]                    # Pa
        IB0 = x[..., 1:2]                   # A
        # n_n = 1e20 * PB + 1e18              # m^-3
        n_n = 900 * PB + 1
        # q = 1.602176634e-19                 # C
        q = 0.25
        A = 0.5
        # A = np.pi * (0.05**2 - 0.035**2)    # m^2
        ui = (IB0/A) / (q*n_n)              # m/s
        y = np.concatenate((n_n, ui), axis=-1)
        return dict(y=y)

    def beam_dump(x):
        Vw = x[..., 0:1]
        ui = x[..., 1:2]
        u_wall = ui - 0.1 * Vw
        gamma_n = 4.5 * np.tanh(u_wall/4 - 2.5) + 5.5
        # u_wall = ui - 10 * Vw   # m/s
        # gamma_n = 1e23 - 4.95e18*u_wall
        return dict(y=gamma_n)

    def chamber_wall(x):
        Vw = x[..., 0:1]    # V
        phi = x[..., 1:2]   # [-]
        ui = x[..., 2:3]    # m/s
        Rs = x[..., 3:4]    # Ohms
        # Rw = phi * (ui / 100)
        Rw = (phi + 0.1) * (ui + 10)
        Iw = Vw / Rw
        Vc = Iw * Rs
        return dict(y=Vc)

    def spacecraft(x):
        T = x[..., 0:1]
        Vc = x[..., 1:2]
        LT_max = 20
        LT_min = 15
        A = (LT_max - LT_min) / 2
        V_max = 60
        f = V_max/4
        LT = A * np.tanh(-2/f * (Vc/100 - V_max/2)) + (LT_max - A)
        Rs = 100 - 8000 * T**2
        y = np.concatenate((Rs, LT), axis=-1)
        return dict(y=y)

    return cathode, thruster, plume, beam_dump, chamber_wall, spacecraft


def chatgpt_model(x):
    y = (np.tanh(x[..., 0] * x[..., 1]) + np.sin(np.pi*x[..., 2] * x[..., 3]) + np.abs(x[..., 4] * x[..., 5]) +
         np.exp(x[..., 6] * x[..., 7]) + x[..., 8]*np.log(np.abs(x[..., 9]) + 1) +
         np.sqrt(np.abs(x[..., 10] * x[..., 11])))

    return {'y': y[..., np.newaxis]}


def chatgpt_system():
    d = 12
    exo_vars = [UniformRV(0, 1, id=f'x{i}') for i in range(d)]
    y = UniformRV(0, 1, 'y0')
    comp = ComponentSpec(chatgpt_model, name='chatgpt', max_beta=(2,)*d)
    surr = SystemSurrogate(comp, exo_vars, y, est_bds=1000, stdout=False)

    return surr
