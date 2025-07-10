![Logo](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/docs/assets/hallmd_logo_text.svg)

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
![build](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/deploy.yml?logo=github)
![docs](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
![Code Coverage](https://img.shields.io/badge/coverage-78%25-yellowgreen?logo=codecov)
[![Journal article](https://img.shields.io/badge/DOI-10.1007/s44205--024--00079--w-blue)](https://rdcu.be/dVmim)

A prototype of a predictive engineering model (PEM) of a Hall thruster.
Integrates sub-models from multiple disciplines to simulate a Hall thruster operating in a vacuum chamber.
Uses uncertainty quantification techniques to extrapolate model predictions to a space-like environment.

## ⚙️ Installation
Ensure you are using Python 3.11 or later. You can install using [pdm](https://github.com/pdm-project/pdm):
```shell
pip install --user pdm
git clone https://github.com/JANUS-Institute/HallThrusterPEM.git  # or download the source from releases
cd HallThrusterPEM
pdm install --prod
```

`hallmd` uses the [`HallThruster.jl`](https://github.com/UM-PEPL/HallThruster.jl) Julia package.
Please see their docs for setting up Julia and installing.
Alternatively, you may run the provided [install script](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/refs/heads/main/scripts/install_hallthruster.py), which will install both Julia and `HallThruster.jl`.
By default, this will install Julia 1.11 and HallThruster.jl version 0.18.7.
From the root directory, run the following command:

```
pdm run scripts/install_hallthruster.py --julia-version=X.XX.X --hallthruster-version=V.VV.V
```

This will create a fresh Julia environment called `hallthruster_V.VV.V` and install `HallThruster.jl` there.

## 📍 Scripts used for publications
See the [scripts](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/scripts) folder for workflows for data generation, parameter inference, and analysis using `hallmd`.
This directory also contains information needed to replicate the results in our publications.

## 📍 Standalone usage

Below, we demonstrate how to use `hallmd` to run HallThruster.jl on a simple config file.

```python
import matplotlib.pyplot as plt

from hallmd.models import hallthruster_jl

config = {
    'discharge_voltage': 300,
    'anode_mass_flow_rate': 5e-6,
    'background_pressure_Torr': 1e-5,
    'propellant': 'Xenon',
    'domain': [0, 0.08]
}

outputs = hallthruster_jl(thruster='SPT-100', config=config)

ion_velocity = outputs['u_ion']
grid = outputs['u_ion_coords']

fig, ax = plt.subplots()
ax.plot(grid, ion_velocity)
ax.set_xlabel('Axial location (m)')
ax.set_ylabel('Ion velocity (m/s)')
plt.show()
```

## 🗂️ Project structure
```tree
HallThrusterPEM
    docs/
    scripts/           # Scripts for building predictive engineering models (PEMs)
        pem_v0/        # PEM v0 coupling of cathode -> thruster -> plume
    src/hallmd         # Python package source code root
        models/        # Python wrappers for sub-models
        data/          # Experimental data
        devices/       # Device information (thrusters, equipment, etc.)
        utils.py       # Utility functions
    tests/             # Testing for Python package
    pdm.lock           # Frozen dependencies file
```

For more info on building PEMs with `hallmd`, see the [scripts](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/scripts).

## 🏗️ Contributing
See the [contribution](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/CONTRIBUTING.md) guidelines.

## 📖 Reference
[[1](https://rdcu.be/dVmim)] Eckels, J. et al., "Hall thruster model improvement by multidisciplinary uncertainty quantification," _Journal of Electric Propulsion_, vol 3, no 19, September 2024.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
