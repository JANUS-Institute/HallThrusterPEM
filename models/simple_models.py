"""Module for simple algebraic models for testing purposes"""
import numpy as np
import sys
import uuid
from pathlib import Path
import pickle

sys.path.append('..')

from utils import NormalRV, UniformRV
from surrogates.system import SystemSurrogate


def custom_nonlinear(x, env_var=0.1**2, wavelength=0.5, wave_amp=0.1, tanh_amp=0.5, L=1, t=0.25):
    """Custom nonlinear model for testing
    Parameters
    ----------
    x: (..., xdim) Input locations
    env_var: Variance of Gaussian envelope
    wavelength: Sinusoidal perturbation wavelength
    wave_amp: Amplitude of perturbation
    tanh_amp: Amplitude of tanh(x)
    L: domain length of underlying tanh function
    t: transition length of tanh function (as fraction of L)

    Returns
    -------
    y: (..., y_dim)
    """
    # Traveling sinusoid with moving Gaussian envelope (theta is x2)
    env_range = [0.2, 0.6]
    mu = env_range[0] + x[..., 1] * (env_range[1] - env_range[0])
    theta_env = 1 / (np.sqrt(2 * np.pi * env_var)) * np.exp(-0.5 * (x[..., 0] - mu) ** 2 / env_var)
    ftheta = wave_amp * np.sin((2*np.pi/wavelength) * x[..., 1]) * theta_env

    # Underlying tanh dependence on x1
    fd = tanh_amp * np.tanh(2/(L*t)*(x[..., 0] - L/2)) + tanh_amp

    # Compute model = f(theta, d) + f(d)
    y = np.expand_dims(ftheta + fd, axis=-1)  # (..., 1)
    return y


def tanh_func(x, A=2, L=1, frac=4):
    """Simple tunable tanh function"""
    return A*np.tanh(2/(L/frac)*(x-L/2)) + A + 1


def ishigami(x, a=7.0, b=0.1):
    """For testing Sobol indices: https://doi.org/10.1109/ISUMA.1990.151285"""
    return np.sin(x[..., 0:1]) + a*np.sin(x[..., 1:2])**2 + b*(x[..., 2:3]**4)*np.sin(x[..., 0:1])


def borehole_func(x, *args, **kwargs):
    """Model found at https://www.sfu.ca/~ssurjano/borehole.html
    :returns vdot: Water flow rate in m^3/yr
    """
    rw = x[..., 0]      # Radius of borehole (m)
    r = x[..., 1]       # Radius of influence (m)
    Tu = x[..., 2]      # Transmissivity (m^2/yr)
    Hu = x[..., 3]      # Potentiometric head (m)
    Tl = x[..., 4]      # Transmissivity (m^2/yr)
    Hl = x[..., 5]      # Potentiometric head (m)
    L = x[..., 6]       # Length of borehole (m)
    Kw = x[..., 7]      # Hydraulic conductivity (m/yr)
    # rw = 0.1
    # r = 10000
    # Tu = 80000
    # Hu = 1000
    # Tl = 90
    # Hl = 750
    # L = 1400
    # Kw = 11000
    vdot = 2*np.pi*Tu*(Hu-Hl) / (np.log(r/rw) * (1 + (2*L*Tu/(np.log(r/rw)*Kw*rw**2)) + (Tu/Tl)))

    return vdot[..., np.newaxis]


def wing_weight_func(x, *args, **kwargs):
    """Model found at https://www.sfu.ca/~ssurjano/wingweight.html
    :returns Wwing: the weight of the airplane wing (lb)
    """
    Sw = x[..., 0]      # Wing area (ft^2)
    Wfw = x[..., 1]     # Weight of fuel (lb)
    A = x[..., 2]       # Aspect ratio
    Lambda = x[..., 3]  # Quarter-chord sweep (deg)
    q = x[..., 4]       # Dynamic pressure (lb/ft^2)
    lamb = x[..., 5]    # taper ratio
    tc = x[..., 6]      # Aerofoil thickness to chord ratio
    Nz = x[..., 7]      # Ultimate load factor
    Wdg = x[..., 8]     # Design gross weight (lb)
    Wp = x[..., 9]      # Paint weight (lb/ft^2)
    # Sw = 175
    # Wfw = 270
    # A = 8
    # Lambda = 1
    # q = 30
    # lamb = 0.7
    # tc = 0.1
    # Nz = 4
    # Wdg = 2000
    # Wp = 0.05
    Lambda = Lambda*(np.pi/180)
    Wwing = 0.036*(Sw**0.758)*(Wfw**0.0035)*((A/(np.cos(Lambda))**2)**0.6)*(q**0.006)*(lamb**0.04)*\
            (100*tc/np.cos(Lambda))**(-0.3)*((Nz*Wdg)**0.49) + Sw*Wp

    return {'y': Wwing[..., np.newaxis]}


def fire_sat_system():
    """Fire satellite detection system model from Chaudhuri 2018"""
    Re = 6378140    # Radius of Earth (m)
    mu = 3.986e14   # Gravitational parameter (m^3 s^-2)
    eta = 0.22      # Power efficiency
    Id = 0.77       # Inherent degradation of the array
    thetai = 0      # Sun incidence angle
    LT = 15         # Spacecraft lifetime (years)
    eps = 0.0375    # Power production degradation (%/year)
    rlw = 3         # Length to width ratio
    nsa = 3         # Number of solar arrays
    rho = 700       # Mass density of arrays (kg/m^3)
    t = 0.005       # Thickness (m)
    D = 2           # Distance between panels (m)
    IbodyX = 6200   # kg*m^2
    IbodyY = 6200   # kg*m^2
    IbodyZ = 4700   # kg*m^2
    dt_slew = 760   # s
    theta = 15      # Deviation of moment axis from vertical (deg)
    As = 13.85      # Area reflecting radiation (m^2)
    c = 2.9979e8    # Speed of light (m/s)
    M = 7.96e15     # Magnetic moment of earth (A*m^2)
    Rd = 5          # Residual dipole of spacecraft (A*m^2)
    rhoa=5.148e-11  # Atmospheric density (kg/m^3) -- typo in Chaudhuri 2018 has this as 1e11 instead
    A = 13.85       # Cross-section in flight (m^2)
    Phold = 20      # Holding power (W)
    omega = 6000    # Max vel of wheel (rpm)
    nrw = 3         # Number of reaction wheels

    def orbit_fun(x, *args, **kwargs):
        H = x[..., 0:1]                         # Altitude (m)
        phi = x[..., 1:2]                       # Target diameter (m)
        vel = np.sqrt(mu / (Re + H))            # Satellite velocity (m/s)
        dt_orbit = 2*np.pi*(Re + H) / vel       # Orbit period (s)
        dt_eclipse = (dt_orbit/np.pi)*np.arcsin(Re / (Re + H))  # Eclipse period (s)
        theta_slew = np.arctan(np.sin(phi / Re) / (1 - np.cos(phi / Re) + H/Re))    # Max slew angle (rad)
        if np.random.rand() < kwargs.get('pct_failure', 0):
            i = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            i2 = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            vel[i + (0,)] = np.nan
            theta_slew[i2 + (0,)] = np.nan
        y = np.concatenate((vel, dt_orbit, dt_eclipse, theta_slew), axis=-1)
        return {'y': y}

    def power_fun(x, *args, output_dir=None, **kwargs):
        pct = 1 - (2 - args[0][0]) * 0.04 if len(args) == 1 else 1
        Po = x[..., 0:1]            # Other power sources (W)
        Fs = x[..., 1:2]            # Solar flux (W/m^2)
        dt_orbit = x[..., 2:3]      # Orbit period (s)
        dt_eclipse = x[..., 3:4]    # Eclipse period (s)
        Pacs = x[..., 4:5]          # Power from attitude control system (W)
        Ptot = Po + Pacs
        Pe = Ptot
        Pd = Ptot
        Xe = 0.6                     # These are power efficiencies in eclipse and daylight
        Xd = 0.8                     # See Ch. 11 of Wertz 1999 SMAD
        Te = dt_eclipse
        Td = dt_orbit - Te
        Psa = ((Pe*Te/Xe) + (Pd*Td/Xd)) / Td
        Pbol = eta*Fs*Id*np.cos(thetai)
        Peol = Pbol * (1 - eps)**LT
        Asa = Psa / Peol            # Total solar array size (m^2)
        L = np.sqrt(Asa*rlw/nsa)
        W = np.sqrt(Asa/(rlw*nsa))
        msa = 2*rho*L*W*t           # Mass of solar array
        Ix = msa*((1/12)*(L**2 + t**2) + (D+L/2)**2) + IbodyX
        Iy = (msa/12)*(L**2 + W**2) + IbodyY  # typo in Zaman 2013 has this as H**2 instead of L**2
        Iz = msa*((1/12)*(L**2 + W**2) + (D + L/2)**2) + IbodyZ
        Itot = np.concatenate((Ix, Iy, Iz), axis=-1)
        Imin = np.min(Itot, axis=-1, keepdims=True)
        Imax = np.max(Itot, axis=-1, keepdims=True)
        if np.random.rand() < kwargs.get('pct_failure', 0):
            i = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            i2 = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            Imin[i2 + (0,)] = np.nan
            Asa[i + (0,)] = np.nan
        y = np.concatenate((Imin, Imax*pct, Ptot*pct, Asa), axis=-1)

        if output_dir is not None:
            files = []
            id = str(uuid.uuid4())
            for index in np.ndindex(*x.shape[:-1]):
                fname = f'{id}_{index}.pkl'
                with open(Path(output_dir) / fname, 'wb') as fd:
                    pickle.dump({'y': y[index + (slice(None),)]}, fd)
                files.append(fname)
            return {'y': y, 'files': files}
        else:
            return {'y': y}

    def attitude_fun(x, *args, output_dir=None, **kwargs):
        pct = 1 - (2 - args[0][0])*0.04 if len(args) == 1 else 1
        H = x[..., 0:1]             # Altitude (m)
        Fs = x[..., 1:2]            # Solar flux
        Lsp = x[..., 2:3]           # Moment arm for solar radiation pressure
        q = x[..., 3:4]             # Reflectance factor
        La = x[..., 4:5]            # Moment arm for aerodynamic drag
        Cd = x[..., 5:6]            # Drag coefficient
        vel = x[..., 6:7]           # Satellite velocity
        theta_slew = x[..., 7:8]    # Max slew angle
        Imin = x[..., 8:9]          # Minimum moment of inertia
        Imax = x[..., 9:10]         # Maximum moment of inertia
        tau_slew = 4*theta_slew*Imax / dt_slew**2
        tau_g = 3*mu*np.abs(Imax - Imin)*np.sin(2*theta*(np.pi/180)) / (2*(Re+H)**3)
        tau_sp = Lsp*Fs*As*(1+q)*np.cos(thetai) / c
        tau_m = 2*M*Rd / (Re + H)**3
        tau_a = (1/2)*La*rhoa*Cd*A*vel**2
        tau_dist = np.sqrt(tau_g**2 + tau_sp**2 + tau_m**2 + tau_a**2)
        tau_tot = np.max(np.concatenate((tau_slew, tau_dist), axis=-1), axis=-1, keepdims=True)
        Pacs = tau_tot*(omega*(2*np.pi/60)) + nrw*Phold
        y = np.concatenate((Pacs, tau_tot), axis=-1) * pct

        if output_dir is not None:
            files = []
            id = str(uuid.uuid4())
            for index in np.ndindex(*x.shape[:-1]):
                fname = f'{id}_{index}.pkl'
                with open(Path(output_dir) / fname, 'wb') as fd:
                    pickle.dump({'y': y[index + (slice(None),)]}, fd)
                files.append(fname)
            return {'y': y, 'files': files}
        else:
            return {'y': y}

    orbit = {'name': 'Orbit', 'model': orbit_fun, 'truth_alpha': (), 'exo_in': [0, 1], 'coupling_in': {},
             'coupling_out': [0, 1, 2, 3], 'max_alpha': (), 'max_beta': (3, 3), 'type': 'lagrange',
             'model_kwargs': {'pct_failure': 0}}
    power = {'name': 'Power', 'model': power_fun, 'truth_alpha': (2,), 'exo_in': [2, 3], 'max_alpha': (2,),
             'coupling_in': {'Orbit': [1, 2], 'Attitude': [0]}, 'coupling_out': [4, 5, 6, 7], 'type': 'lagrange',
             'max_beta': (3,)*5, 'save_output': True, 'model_kwargs': {'pct_failure': 0}}
    attitude = {'name': 'Attitude', 'model': attitude_fun, 'truth_alpha': (2,), 'exo_in': [0, 3, 4, 5, 6, 7],
                'max_alpha': (2,), 'coupling_in': {'Orbit': [0, 3], 'Power': [0, 1]}, 'coupling_out': [8, 9],
                'type': 'lagrange', 'max_beta': (3,)*10, 'save_output': True}
    exo_vars = [NormalRV(18e6, 1e6, id='H'), NormalRV(235e3, 10e3, id='\u03D5'), NormalRV(1000, 50, id='Po'),
                NormalRV(1400, 20, id='Fs'), NormalRV(2, 0.4, id='Lsp'), NormalRV(0.5, 0.1, id='q'),
                NormalRV(2, 0.4, id='La'), NormalRV(1, 0.2, id='Cd')]
    coupling_vars = [UniformRV(2000, 6000, id='Vsat'), UniformRV(20000, 60000, id='To'), UniformRV(1000, 5000, id='Te'),
                     UniformRV(0, 4, id='Slew'), UniformRV(0, 12000, id='Imin'), UniformRV(0, 12000, id='Imax'),
                     UniformRV(0, 10000, id='Ptot'), UniformRV(0, 50, id='Asa'), UniformRV(0, 100, id='Pat'),
                     UniformRV(0, 5, id='tau_tot')]
    sys = SystemSurrogate([orbit, power, attitude], exo_vars, coupling_vars, est_bds=500)
    return sys


def fake_pem():
    """Simple algebraic models for mocking a Hall thruster PEM"""
    def cathode(x, alpha):
        PB = x[..., 0:1]
        V_vac = 30
        Te = 2
        PT = 4e-06*133.33
        Pstar = 3e-05*133.33
        Vcc = V_vac + Te * np.log(1 + PB / PT) - (Te / (PT + Pstar)) * PB
        return Vcc

    def thruster(x, alpha):
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
        return y

    def plume(x, alpha):
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
        return y

    def beam_dump(x, alpha):
        Vw = x[..., 0:1]
        ui = x[..., 1:2]
        u_wall = ui - 0.1 * Vw
        gamma_n = 4.5 * np.tanh(u_wall/4 - 2.5) + 5.5
        # u_wall = ui - 10 * Vw   # m/s
        # gamma_n = 1e23 - 4.95e18*u_wall
        return gamma_n

    def chamber_wall(x, alpha):
        Vw = x[..., 0:1]    # V
        phi = x[..., 1:2]   # [-]
        ui = x[..., 2:3]    # m/s
        Rs = x[..., 3:4]    # Ohms
        # Rw = phi * (ui / 100)
        Rw = (phi + 0.1) * (ui + 10)
        Iw = Vw / Rw
        Vc = Iw * Rs
        return Vc

    def spacecraft(x, alpha):
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
        return y

    return cathode, thruster, plume, beam_dump, chamber_wall, spacecraft


def borehole_system():
    d = 8
    idx = list(np.arange(d))
    vars = [UniformRV(0.05, 0.15, 'rw'), UniformRV(100, 50000, 'r'), UniformRV(63070, 115600, 'Tu'),
            UniformRV(990, 1110, 'Hu'), UniformRV(63.1, 116, 'Tl'), UniformRV(700, 820, 'Hl'),
            UniformRV(1120, 1680, 'L'), UniformRV(9855, 12045, 'Kw')]
    coupling_vars = [UniformRV(0, 1000, 'vdot')]
    exo_vars = [vars[i] for i in idx]
    comp = {'name': 'Borehole', 'model': borehole_func, 'exo_in': idx, 'local_in': {}, 'global_out': [0],
            'truth_alpha': (), 'max_beta': (3,)*d}
    sys = SystemSurrogate([comp], exo_vars, coupling_vars, root_dir='build', stdout=False, est_bds=1000)

    return sys


def wing_weight_system():
    d = 10
    idx = [int(i) for i in np.arange(d)]
    vars = [UniformRV(150, 250, id='Sw'), UniformRV(220, 300, id='Wfw'), UniformRV(6, 10, id='A'),
            UniformRV(-10, 10, id='Lambda'), UniformRV(16, 45, id='q'), UniformRV(0.5, 1, id='lambda'),
            UniformRV(0.08, 0.18, id='tc'), UniformRV(2.5, 6, id='Nz'), UniformRV(1700, 2500, id='Wdg'),
            UniformRV(0.025, 0.08, id='Wp')]
    coupling_vars = [UniformRV(0, 10000, id='Wwing')]
    exo_vars = [vars[i] for i in idx]
    comp = {'name': 'Wing', 'model': wing_weight_func, 'exo_in': idx, 'coupling_in': {}, 'coupling_out': [0],
            'truth_alpha': (), 'max_beta': (2,)*d, 'type': 'lagrange'}
    sys = SystemSurrogate([comp], exo_vars, coupling_vars, root_dir='build', est_bds=1000)

    return sys


def chatgpt_model(x, *args, **kwargs):
    y = (np.tanh(x[..., 0] * x[..., 1]) + np.sin(np.pi*x[..., 2] * x[..., 3]) + np.abs(x[..., 4] * x[..., 5]) +
         np.exp(x[..., 6] * x[..., 7]) + x[..., 8]*np.log(np.abs(x[..., 9]) + 1) +
         np.sqrt(np.abs(x[..., 10] * x[..., 11])))

    return {'y': y[..., np.newaxis]}


def chatgpt_system():
    d = 12
    exo_vars = [UniformRV(0, 1, id=f'x{i}') for i in range(d)]
    exo_idx = [int(i) for i in range(d)]
    coupling_vars = [UniformRV(0, 1, id='y0')]
    comp = {'name': 'chatgpt', 'model': chatgpt_model, 'exo_in': exo_idx, 'coupling_in': {}, 'coupling_out': [0],
            'truth_alpha': (), 'max_beta': (2,)*d, 'type': 'lagrange'}
    surr = SystemSurrogate([comp], exo_vars, coupling_vars, root_dir='build', est_bds=1000)

    return surr
