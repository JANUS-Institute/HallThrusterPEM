![Logo](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/docs/assets/hallmd_logo_text.svg)

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
![build](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/deploy.yml?logo=github)
![docs](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/JANUS-Institute/HallThrusterPEM/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
![Code Coverage](https://img.shields.io/badge/coverage-89%25-yellowgreen?logo=codecov)
[![Journal article](https://img.shields.io/badge/DOI-10.1007/s44205--024--00079--w-blue)](https://rdcu.be/dVmim)

Prototype of a predictive engineering model (PEM) of a Hall thruster. Integrates sub-models from multiple disciplines to simulate a Hall thruster operating in a vacuum chamber. Uses uncertainty quantification techniques to extrapolate model predictions to a space-like environment.

## ⚙️ Installation
Using [pdm](https://github.com/pdm-project/pdm):
```shell
pip install --user pdm
git clone https://github.com/JANUS-Institute/HallThrusterPEM.git  # or download the source from releases
cd HallThrusterPEM
pdm install --prod
```

`hallmd` uses the [`HallThruster.jl`](https://github.com/UM-PEPL/HallThruster.jl) Julia package. Please see their docs for setting up Julia and installing. Alternatively, you may run the provided [install script](https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/install_hallthruster.py), which will install both Julia and `HallThruster.jl`.

Assuming `python` is available on your path:

=== Linux/Mac
    ```shell
    curl -sSL https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/install_hallthruster.py | python -
    ```

=== Windows
    ```shell
    powershell -c "Invoke-WebRequest -Uri https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/main/install_hallthruster.py | python -"
    ```

## 📍 Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt

from hallmd.models.pem import pem_v0
from hallmd.data.loader import spt100_data
from hallmd.utils import plot_qoi

system = pem_v0()
system.fit(max_iter=10)

# Show model predictions vs experimental thrust data
data = spt100_data(['T'])[0]
inputs = data['x']  # Pressure, Anode voltage, Anode mass flow rate
Nx, num_samples = inputs.shape[0], 100
xs = np.zeros((Nx, num_samples, len(system.exo_vars)))

for i in range(Nx):
    nominal = dict(PB=inputs[i, 0], Va=inputs[i, 1], mdot_a=inputs[i, 2])
    xs[i, :, :] = system.sample_inputs(num_samples, use_pdf=True, nominal=nominal)
    
ys = system.predict(xs, qoi_ind='T')*1000       # Predicted thrust [mN]
y = data['y']*1000                              # Experimental thrust [mN]
y_err = 2 * np.sqrt(data['var_y'])*1000         # Experimental noise [mN]

fig, ax = plt.subplots()
pressure = 10 ** data[:, 0]
idx = np.argsort(pressure)
ax.errorbar(pressure, y, yerr=y_err, fmt='or', capsize=3, label='Experiment')
plot_qoi(ax, pressure[idx], ys[idx, :], 'Background pressure (Torr)', 'Thrust (mN)', legend=True)
plt.show()
```

## Project structure
```tree
HallThrusterPEM                 # Root project directory
|- docs                         # Documentation and references
|- scripts                      # Scripts for building PEM surrogates
|  |- pem_v0                    # PEM v0 surrogate scripts
|  |  |- gen_data.sh
|  |  |- train_surr.sh
|  |- debug                     # Scripts for debugging SLURM workflow
|  |- analysis                  # Scripts for UQ analysis (Monte Carlo, Sobol, etc.)
|  |- ...
|- src/hallmd                   # Python package source code root
|  |- models                    # Python wrappers for sub-models
|  |  |- thruster.py
|  |  |- ...
|  |- data                      # Experimental data
|  |  |- spt100                 # Contains all data for the SPT-100
|  |  |- ...
|  |  |- loader.py              # Helper functions for loading data
|  |- utils.py                  # Useful utility functions
|  |- juliapkg.json             # Specifies version of Hallthruster.jl
|- tests                        # Testing for models, generating data, and plotting results
|- results                      # Test scripts write data here (but kept out of the repo)
|- pdm.lock                     # Frozen dependencies file
|- setup_env.sh                 # Convenience script for setting up pdm environment
```

## 🏗️ Contributing
See the [contribution](https://github.com/JANUS-Institute/HallThrusterPEM/blob/main/CONTRIBUTING.md) guidelines.

## 📖 Reference
[[1](https://rdcu.be/dVmim)] Eckels, J. et al., "Hall thruster model improvement by multidisciplinary uncertainty quantification," Journal of Electric Propulsion, September 2024.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
