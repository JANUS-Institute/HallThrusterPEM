"""All models are specified as callable functions in `hallmd.models`. Currently supported models are based on
a three-component feedforward system for a Hall thruster:

1. **Cathode** - Accounts for interactions of the cathode plasma with the main discharge.
2. **Thruster** - The primary simulation of the Hall thruster channel discharge and near-field.
3. **Plume** - Models the far-field expansion of the plasma plume in the vacuum chamber.

![The three-component feedforward Hall thruster model](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/refs/heads/main/docs/assets/pem_v0.svg)

**Fig 1.** The three-component feedforward Hall thruster model (Eckels et al 2024).

Examples of integrated predictive engineering models (PEM) are included in the
[scripts](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/scripts) folder.
"""
from .cathode import cathode_coupling
from .plume import current_density
from .thruster import hallthruster_jl

__all__ = ['cathode_coupling', 'hallthruster_jl', 'current_density']
