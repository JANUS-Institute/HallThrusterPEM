![Logo](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/docs/assets/hallmd_logo_text.svg)

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
![build](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/deploy.yml?logo=github)
![docs](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
![Code Coverage](https://img.shields.io/badge/coverage-77%25-yellowgreen?logo=codecov)
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
From the root directory, run the following command (assuming `python` is available on your path)

### Linux/Mac
```shell
curl -sSL https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/refs/heads/main/scripts/install_hallthruster.py | python -
```

### Windows
```shell
powershell -c "Invoke-WebRequest -Uri https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/refs/heads/main/scripts/install_hallthruster.py | python -"
```
This will create a fresh Julia environment called `hallthruster_<VERSION>` (where `<VERSION>` is the latest version of HallThruster.jl) and install `HallThruster.jl` there.
You can also use `pdm` to run this workflow and specify a specific julia and HallThruster.jl version.

```shell
pdm run scripts/install_hallthruster.py --julia-version=X.XX.X --hallthruster-version=V.VV.V
```

## 📍 Scripts used for publications
See the [scripts](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/scripts) folder for workflows for data generation, parameter inference, and analysis using `hallmd`.
This directory also contains information needed to replicate the results in our publications.

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

## Funding declaration

Funding for this work was provided by NASA in part through the [Joint Advanced Propulsion Institute (JANUS)](https://januselectricpropulsion.com/), a NASA Space Technology Research Institute, under grant number 80NSSC21K1118, as well as in part through a NASA Space Technology Graduate Research Opportunity grant 80NSSC23K1181.
This research was additionally supported in part through computational resources provided by [Advanced Research Computing](https://its.umich.edu/advanced-research-computing) at the University of Michigan.

## 📖 References
[[1](https://rdcu.be/dVmim)] Eckels, J. et al., "Hall thruster model improvement by multidisciplinary uncertainty quantification," _Journal of Electric Propulsion_, vol 3, no 19, September 2024.
[[2](https://rdcu.be/dVmim)] Marks, T. et al., "Uncertainty quantification of a multi-component Hall thruster model at varying facility pressures". _Journal of Applied Physics_, 138(15). October 2025.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
