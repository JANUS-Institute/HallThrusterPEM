""" `utils.py`

Module to provide utilities for the `hallmd` package.

Includes
--------
- `ModelRunException` - Used to note when a model has encountered an unknown error.
- `data_write()` - Convenience function for writing .json data to file.
- `plot_qoi()` - Convenience plotting tool for showing QoI with UQ bounds
"""
import json
from pathlib import Path
from importlib import resources

import numpy as np
from uqtils import ax_default

FUNDAMENTAL_CHARGE = 1.602176634e-19   # Fundamental charge (C)
BOLTZMANN_CONSTANT = 1.380649e-23      # Boltzmann constant (J/K)
TORR_2_PA = 133.322                    # Conversion factor from Torr to Pa


def model_config_dir():
    """Return a path to the model configuration directory"""
    return resources.files('hallmd.models.config')


def plot_qoi(ax, x, qoi, xlabel, ylabel, legend=False):
    """ Plot a quantity of interest with 5%, 50%, 95% percentiles against `x`.

    :param ax: matplotlib Axes object to plot on
    :param x: `(Nx,)` array to plot on `x` axis
    :param qoi: `(Nx, Ns,)` samples of the QOI at each `x` location
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param legend: whether to plot a legend
    """
    p5 = np.percentile(qoi, 5, axis=1)
    med = np.percentile(qoi, 50, axis=1)
    p95 = np.percentile(qoi, 95, axis=1)
    ax.plot(x, med, '-k', label='Model')
    ax.fill_between(x, p5, p95, alpha=0.4, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.8, 0.8, 0.8))
    ax_default(ax, xlabel, ylabel, legend=legend)


class ModelRunException(Exception):
    """Custom exception to note when a model has encountered an unknown error while executing."""
    pass


def data_write(data, filename, write_dir='.'):
    """Convenience function to write .json data files."""
    with open(Path(write_dir) / filename, 'w', encoding='utf-8') as fd:
        json.dump(data, fd, ensure_ascii=False, indent=4)
