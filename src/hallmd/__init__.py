"""Prototype of a predictive engineering model (PEM) of a Hall thruster. Integrates sub-models from multiple
disciplines to simulate a Hall thruster operating in a vacuum chamber. Uses uncertainty quantification techniques
to extrapolate model predictions to a space-like environment.

- Authors - Joshua Eckels (eckelsjd@umich.edu), Thomas Marks, Madison Allen, Declan Brick, Benjamin Jorns,
Alex Gorodetsky
- License - GPL-3.0

The `hallmd` package contains three sub-packages:

- `models` - Contains the sub-models for each discipline in the Hall thruster system.
- `devices` - Contains information about specific devices (thrusters and other equipment).
- `data` - Contains experimental data for validating the models.
"""

import numpy as _np

__version__ = '0.2.0'
