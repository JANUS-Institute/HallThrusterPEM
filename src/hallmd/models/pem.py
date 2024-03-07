""" `pem.py`

Module for full multidisciplinary Hall thruster predictive engineering model(s).

!!! Note
    Multidisciplinary systems are specified using the `SystemSurrogate` object from the `amisc` package. This data
    structure allows feedforward and feedback connections between a set of component models. It can be used to build
    a surrogate for the MD system, or to evaluate the system directly using the underlying component models.

Includes
--------
- `pem_v0()` - The v0 cathode-thruster-plume feedforward multidisciplinary system.
"""
from pathlib import Path
import pickle
from concurrent.futures import Executor

from amisc.system import ComponentSpec, SystemSurrogate
from amisc.rv import UniformRV
from amisc.utils import load_variables

from hallmd.models.cc import cc_feedforward
from hallmd.models.thruster import hallthruster_jl_wrapper
from hallmd.models.plume import plume_feedforward
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def pem_v0(save_dir: str | Path = None, executor: Executor = None, init: bool = True,
           hf_override: bool = False, var_file: str | Path = CONFIG_DIR / 'variables_v0.json',
           from_file: str | Path = None) -> SystemSurrogate:
    """Return a `SystemSurrogate` object for the feedforward v0 PEM system.

    :param save_dir: where to save surrogate and model outputs
    :param executor: the parallel execution manager
    :param init: whether to initialize the surrogate (will evaluate all component models)
    :param hf_override: whether to use only highest-fidelity for all models
    :param var_file: the path to the `.json` config file storing information about all variables
    :param from_file: the `.pkl` save file to load the surrogate from, (instead of building from scratch)
    :returns: the `SystemSurrogate` object
    """
    exo_vars = load_variables(['PB', 'Va', 'mdot_a', 'T_ec', 'V_vac', 'P*', 'PT', 'u_n', 'l_t', 'vAN1', 'vAN2',
                               'delta_z', 'z0', 'p0', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'sigma_cex', 'r_m'],
                              var_file)
    coupling_vars = load_variables(['V_cc', 'I_B0', 'I_D', 'T', 'eta_v', 'eta_c', 'eta_m', 'ui_avg', 'theta_d'], var_file)

    if from_file is not None:
        surr = SystemSurrogate.load_from_file(Path(from_file), stdout=False, executor=executor)
        for v in exo_vars:
            # Make sure nominal values are up to date with the current config file
            j = surr.exo_vars.index(v)
            surr.exo_vars[j].nominal = v.nominal
            surr.exo_vars[j].param_type = v.param_type
            surr.exo_vars[j].tex = v.tex
            surr.exo_vars[j].update_bounds(*v.bounds())

        return surr

    # Get number of reconstruction coefficients for ion velocity and ion current density profiles
    try:
        with open(CONFIG_DIR / 'thruster_svd.pkl', 'rb') as fd, open(CONFIG_DIR / 'plume_svd.pkl', 'rb') as fd2:
            d = pickle.load(fd)
            r1 = d['vtr'].shape[0]
            d2 = pickle.load(fd2)
            r2 = d2['vtr'].shape[0]
    except FileNotFoundError:
        r1 = 11
        r2 = 6

    coupling_vars.extend([UniformRV(-20, 20, id=f'uion{i}', tex=f"$\\tilde{{u}}_{{ion,{i}}}$",
                                    description=f'Ion velocity latent coefficient {i}',
                                    param_type='coupling') for i in range(r1)])
    coupling_vars.extend([UniformRV(-20, 20, id=f'jion{i}', tex=f"$\\tilde{{j}}_{{ion,{i}}}$",
                                    description=f'Current density latent coefficient {i}',
                                    param_type='coupling') for i in range(r2)])

    # Component inputs
    cathode_exo = ['PB', 'Va', 'T_ec', 'V_vac', 'P*', 'PT']
    thruster_exo = ['PB', 'Va', 'mdot_a', 'T_ec', 'u_n', 'l_t', 'vAN1', 'vAN2', 'delta_z', 'z0', 'p0']
    plume_exo = ['PB', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'sigma_cex', 'r_m']

    # Models should be specified at the global scope for pickling
    cathode = ComponentSpec(cc_feedforward, name='Cathode', exo_in=cathode_exo, coupling_out='V_cc',
                            surrogate='analytical')
    thruster = ComponentSpec(hallthruster_jl_wrapper, name='Thruster', exo_in=thruster_exo, truth_alpha=(2, 2),
                             max_alpha=(2, 2), coupling_in='V_cc', max_beta=(2,)*(len(thruster_exo)+1), save_output=True,
                             coupling_out=['I_B0', 'I_D', 'T', 'eta_v', 'eta_c', 'eta_m', 'ui_avg'] + [f'uion{i}' for i in range(r1)],
                             model_kwargs=dict(n_jobs=-1, compress=True, hf_override=hf_override))
    plume = ComponentSpec(plume_feedforward, name='Plume', exo_in=plume_exo, coupling_in='I_B0', surrogate='analytical',
                          coupling_out=['theta_d'] + [f'jion{i}' for i in range(r2)], model_kwargs={'compress': True})

    logger_name = 'SF-surrogate' if hf_override else 'MF-surrogate'
    surr = SystemSurrogate([cathode, thruster, plume], exo_vars, coupling_vars, executor=executor, init_surr=init,
                           stdout=False, save_dir=save_dir, logger_name=logger_name)

    return surr
