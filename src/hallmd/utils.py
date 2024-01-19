""" `utils.py`

Module to provide utilities for the `hallmd` package.

Includes
--------
- `ModelRunException` - Used to note when a model has encountered an unknown error.
- `data_write()` - Convenience function for writing .json data to file.
- `plot_qoi()` - Convenience plotting tool for showing QoI with UQ bounds
- `plot_slice()` - Plotting tool for showing 1d slices of a function over inputs
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from amisc.utils import ax_default


class ModelRunException(Exception):
    """Custom exception to note when a model has encountered an unknown error while executing."""
    pass


def data_write(data, filename, write_dir='.'):
    """Convenience function to write .json data files."""
    with open(Path(write_dir) / filename, 'w', encoding='utf-8') as fd:
        json.dump(data, fd, ensure_ascii=False, indent=4)


def plot_qoi(ax, x, qoi, xlabel, ylabel, legend=False):
    """ Plot QOI with 5%, 50%, 95% percentiles against `x`.
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


def plot_slice(fun: callable, bds: list[tuple], x0: list | np.ndarray = None, x_idx: list[int] = None,
               y_idx: list[int] = None, N: int = 50, random_walk: bool = False, xlabels: list[str] = None,
               ylabels: list[str] = None):
    """Helper function to plot 1d slices of a function over inputs.

    :param fun: function callable as `y=f(x)`, with `x` as `(..., xdim)` and `y` as `(..., ydim)`
    :param bds: list of tuples of `(min, max)` specifying the bounds of the inputs
    :param x0: the default values for all inputs; defaults to middle of `bds`
    :param x_idx: list of input indices to take 1d slices of
    :param y_idx: list of output indices to plot 1d slices of
    :param N: the number of points to take in each 1d slice
    :param random_walk: whether to slice in a random d-dimensional direction or hold all params const while slicing
    :param xlabels: list of labels for the inputs
    :param ylabels: list of labels for the outputs
    :returns: `fig, ax` with `num_inputs` by `num_outputs` subplots
    """
    x_idx = list(np.arange(0, min(3, len(bds)))) if x_idx is None else x_idx
    y_idx = [0] if y_idx is None else y_idx
    xlabels = [f'x{i}' for i in range(len(x_idx))] if xlabels is None else xlabels
    ylabels = [f'QoI {i}' for i in range(len(y_idx))] if ylabels is None else ylabels
    x0 = [(b[0] + b[1]) / 2 for b in bds] if x0 is None else x0
    x0 = np.atleast_1d(x0)
    xdim = x0.shape[0]
    lb = np.atleast_1d([b[0] for b in bds])
    ub = np.atleast_1d([b[1] for b in bds])

    # Construct sliced inputs
    xs = np.zeros((N, len(x_idx), xdim))
    for i in range(len(x_idx)):
        if random_walk:
            # Make a random straight-line walk across d-cube
            r0 = np.random.rand(xdim) * (ub - lb) + lb
            r0[x_idx[i]] = lb[x_idx[i]]                     # Start slice at this lower bound
            rf = np.random.rand(xdim) * (ub - lb) + lb
            rf[x_idx[i]] = ub[x_idx[i]]                     # Slice up to this upper bound
            xs[0, i, :] = r0
            for k in range(1, N):
                xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
        else:
            # Otherwise, only slice one variable
            for j in range(xdim):
                if j == x_idx[i]:
                    xs[:, i, j] = np.linspace(lb[x_idx[i]], ub[x_idx[i]], N)
                else:
                    xs[:, i, j] = x0[j]

    # Compute function values and show ydim by xdim grid of subplots
    ys = fun(xs)
    if ys.shape == (N, len(x_idx)):
        ys = ys[..., np.newaxis]

    fig, axs = plt.subplots(len(y_idx), len(x_idx), sharex='col', sharey='row')
    for i in range(len(y_idx)):
        for j in range(len(x_idx)):
            if len(y_idx) == 1:
                ax = axs if len(x_idx) == 1 else axs[j]
            elif len(x_idx) == 1:
                ax = axs if len(y_idx) == 1 else axs[i]
            else:
                ax = axs[i, j]
            x = xs[:, j, x_idx[j]]
            y = ys[:, j, y_idx[i]]
            ax.plot(x, y, '-k')
            ylabel = ylabels[i] if j == 0 else ''
            xlabel = xlabels[j] if i == len(y_idx) - 1 else ''
            # legend = (i == 0 and j == len(x_idx) - 1)
            legend = False
            ax_default(ax, xlabel, ylabel, legend=legend)
    fig.set_size_inches(3 * len(x_idx), 3 * len(y_idx))
    fig.tight_layout()

    return fig, axs
