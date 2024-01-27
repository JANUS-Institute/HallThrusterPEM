""" `utils.py`

Module to provide utilities for the `hallmd` package.

Includes
--------
- `ModelRunException` - Used to note when a model has encountered an unknown error.
- `data_write()` - Convenience function for writing .json data to file.
- `plot_qoi()` - Convenience plotting tool for showing QoI with UQ bounds
- `plot_slice()` - Plotting tool for showing 1d slices of a function over inputs
- `is_positive_definite()` - Used to determine if a matrix is PSD
- `nearest_positive_definite()` - Used to get nearest PSD matrix
"""
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from amisc.utils import ax_default, batch_normal_sample
from numpy.linalg import LinAlgError
import tqdm
import h5py


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


def plot_slice(funs, bds: list[tuple], x0: list | np.ndarray = None, x_idx: list[int] = None,
               y_idx: list[int] = None, N: int = 50, random_walk: bool = False, xlabels: list[str] = None,
               ylabels: list[str] = None, cmap='viridis', fun_labels=None):
    """Helper function to plot 1d slices of a function(s) over inputs.

    :param funs: function callable as `y=f(x)`, with `x` as `(..., xdim)` and `y` as `(..., ydim)`, can also be a list
                of functions to evaluate and plot together
    :param bds: list of tuples of `(min, max)` specifying the bounds of the inputs
    :param x0: the default values for all inputs; defaults to middle of `bds`
    :param x_idx: list of input indices to take 1d slices of
    :param y_idx: list of output indices to plot 1d slices of
    :param N: the number of points to take in each 1d slice
    :param random_walk: whether to slice in a random d-dimensional direction or hold all params const while slicing
    :param xlabels: list of labels for the inputs
    :param ylabels: list of labels for the outputs
    :param cmap: the name of the matplotlib colormap to use
    :param fun_labels: the legend labels if plotting multiple functions on each plot
    :returns: `fig, ax` with `num_inputs` by `num_outputs` subplots
    """
    funs = funs if isinstance(funs, list) else [funs]
    x_idx = list(np.arange(0, min(3, len(bds)))) if x_idx is None else x_idx
    y_idx = [0] if y_idx is None else y_idx
    xlabels = [f'x{i}' for i in range(len(x_idx))] if xlabels is None else xlabels
    ylabels = [f'QoI {i}' for i in range(len(y_idx))] if ylabels is None else ylabels
    fun_labels = [f'fun {i}' for i in range(len(funs))] if fun_labels is None else fun_labels
    x0 = [(b[0] + b[1]) / 2 for b in bds] if x0 is None else x0
    x0 = np.atleast_1d(x0)
    xdim = x0.shape[0]
    lb = np.atleast_1d([b[0] for b in bds])
    ub = np.atleast_1d([b[1] for b in bds])
    cmap = plt.get_cmap(cmap)

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
    ys = []
    for func in funs:
        y = func(xs)
        if y.shape == (N, len(x_idx)):
            y = y[..., np.newaxis]
        ys.append(y)
    c_intervals = np.linspace(0, 1, len(ys))

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
            for k in range(len(ys)):
                y = ys[k][:, j, y_idx[i]]
                ax.plot(x, y, ls='-', color=cmap(c_intervals[k]), label=fun_labels[k])
            ylabel = ylabels[i] if j == 0 else ''
            xlabel = xlabels[j] if i == len(y_idx) - 1 else ''
            legend = (i == 0 and j == len(x_idx) - 1 and len(ys) > 1)
            ax_default(ax, xlabel, ylabel, legend=legend)
    fig.set_size_inches(3 * len(x_idx), 3 * len(y_idx))
    fig.tight_layout()

    return fig, axs


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky."""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except LinAlgError:
        return False


def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def batch_normal_pdf(x, mean, cov, logpdf=False):
    """Compute the normal pdf at each `x` location.

    :param x: `(..., dim)`, the locations to evaluate the pdf at
    :param mean: `(..., dim)`, expected values, where dim is the random variable dimension
    :param cov: `(..., dim, dim)`, covariance matrices
    :returns: `(...,)` the pdf values at `x`
    """
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)
    dim = cov.shape[-1]

    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))
    diff = x - mean
    diff_col = np.expand_dims(diff, axis=-1)    # (..., dim, 1)
    diff_row = np.expand_dims(diff, axis=-2)    # (..., 1, dim)
    inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))

    pdf = np.log(preexp) - 0.5 * inexp if logpdf else preexp * np.exp(-0.5 * inexp)

    return pdf


def dram(logpdf, x0, cov0, niter, gamma=0.5, eps=1e-7, adapt_after=100, delayed=True, progress=True, filename=None):
    """ Delayed adaptive metropolis-hastings MCMC with a Gaussian proposal.

    :param logpdf: log PDF of target distribution
    :param x0: `(nwalkers, ndim)` initial parameter samples
    :param cov0: `(ndim, ndim)` the initial proposal covariance
    :param niter: number of iterations
    :param gamma: scale factor for the covariance matrix for delayed rejection step
    :param eps: small constant for making sure covariance is well-conditioned
    :param adapt_after: the number of iterations before covariance adaptation begins (ignored if <=0)
    :param delayed: whether to try to sample again after rejection
    :param progress: whether to display progress of the sampler
    :param filename: if specified, an hdf5 file to save results to. If the file already has dram results, the new
                     samples will be appended
    :returns: `samples, log_pdf, acceptance` - `(niter, nwalkers, ndim)` samples of the target distribution, the logpdf
              values at these locations, and the cumulative number of accepted samples per walker
    """
    # Initialize
    x0 = np.atleast_2d(x0)
    nwalk, ndim = x0.shape
    sd = (2.4**2/ndim)
    curr_cov = np.broadcast_to(cov0, (nwalk, ndim, ndim)).copy()
    curr_mean = x0
    curr_loc_logpdf = logpdf(x0)
    samples = np.empty((niter, nwalk, ndim), dtype=x0.dtype)
    log_pdf = np.empty((niter, nwalk), dtype=x0.dtype)
    accepted = np.zeros((nwalk,), dtype=x0.dtype)
    samples[0, ...] = x0
    log_pdf[0, ...] = curr_loc_logpdf

    def accept_first(curr_log, prop_log):
        with np.errstate(over='ignore'):
            # Overflow values go to -> infty, so they will always get accepted
            ret = np.minimum(1, np.exp(prop_log - curr_log))
        return ret

    # Main sample loop
    iterable = tqdm.tqdm(range(niter-1)) if progress else range(niter-1)
    for i in iterable:
        # if not is_positive_definite(curr_cov):
        #     print(f'Caught non-positive definite matrix! Fixing...')
        #     curr_cov = nearest_positive_definite(curr_cov)

        # Propose sample
        x1 = samples[i, ...]
        y1 = batch_normal_sample(x1, curr_cov)      # (nwalkers, ndim)
        x1_log = curr_loc_logpdf
        y1_log = logpdf(y1)

        # Compute first acceptance
        a1 = accept_first(x1_log, y1_log)           # (nwalkers,)
        a1_idx = np.random.rand() < a1
        samples[i + 1, a1_idx, :] = y1[a1_idx, :]
        samples[i + 1, ~a1_idx, :] = x1[~a1_idx, :]
        curr_loc_logpdf[a1_idx] = y1_log[a1_idx]
        accepted[a1_idx] += 1

        # Second level proposal
        if delayed and np.any(~a1_idx):
            y2 = batch_normal_sample(x1[~a1_idx, :], curr_cov[~a1_idx, ...] * gamma)
            y2_log = logpdf(y2)
            frac_1 = y2_log - x1_log[~a1_idx]
            frac_2 = (batch_normal_pdf(y1[~a1_idx, :], y2, curr_cov[~a1_idx, ...], logpdf=True) -
                      batch_normal_pdf(y1[~a1_idx, :], x1[~a1_idx, :], curr_cov[~a1_idx, ...], logpdf=True))
            with np.errstate(divide='ignore'):
                # If a(y2, y1)=1, then log(1-a(y2,y1)) -> -infty and a2 -> 0
                frac_3 = np.log(1 - accept_first(y2_log, y1_log[~a1_idx])) - np.log(1 - a1[~a1_idx])
            a2 = np.minimum(1, np.exp(frac_1 + frac_2 + frac_3))
            a2_idx = np.random.rand() < a2

            sample_a2_idx = np.where(~a1_idx)[0][a2_idx]  # Indices that were False the 1st time, then true the 2nd
            samples[i + 1, sample_a2_idx, :] = y2[a2_idx, :]
            curr_loc_logpdf[sample_a2_idx] = y2_log[a2_idx]
            accepted[sample_a2_idx] += 1

        log_pdf[i+1, ...] = curr_loc_logpdf

        # Update the sample mean every iteration
        if abs(adapt_after) > 0:
            last_mean = curr_mean.copy()
            curr_mean = (1/(i+1)) * x1 + (i/(i+1))*last_mean

            if i >= adapt_after:
                k = i
                mult = (np.eye(ndim) * eps + k * last_mean[..., np.newaxis] @ last_mean[..., np.newaxis, :] -
                        (k + 1) * curr_mean[..., np.newaxis] @ curr_mean[..., np.newaxis, :] +
                        x1[..., np.newaxis] @ x1[..., np.newaxis, :])
                curr_cov = ((k - 1) / k) * curr_cov + (sd / k) * mult

    try:
        if filename is not None:
            with h5py.File(filename, 'a') as fd:
                group = fd.get('mcmc', None)
                if group is not None:
                    samples = np.concatenate((group['chain'], samples), axis=0)
                    log_pdf = np.concatenate((group['log_pdf'], log_pdf), axis=0)
                    accepted += group['accepted']
                    del group['chain']
                    del group['log_pdf']
                    del group['accepted']
                fd.create_dataset('mcmc/chain', data=samples)
                fd.create_dataset('mcmc/log_pdf', data=log_pdf)
                fd.create_dataset('mcmc/accepted', data=accepted)
    except Exception as e:
        warnings.warn(str(e))

    return samples, log_pdf, accepted


def autocorrelation(samples, maxlag=100, step=1):
    """Compute the auto-correlation of a set of samples

    :param samples: (niter, nwalk, ndim)
    :param maxlag: maximum distance to compute the correlation for
    :param step: step between distances from 0 to `maxlag` for which to compute the correlations
    :returns: lags, autos, tau, ess - the lag times, auto-correlations, integrated auto-correlation,
              and effective sample sizes
    """
    niter, nwalk, ndim = samples.shape
    mean = np.mean(samples, axis=0)
    var = np.sum((samples - mean[np.newaxis, ...]) ** 2, axis=0)

    lags = np.arange(0, maxlag, step)
    autos = np.zeros((len(lags), nwalk, ndim))
    for zz, lag in enumerate(lags):
        # compute the covariance between all samples *lag apart*
        for ii in range(niter - lag):
            autos[zz, ...] += (samples[ii, ...] - mean) * (samples[ii + lag, ...] - mean)
        autos[zz, ...] /= var
    tau = 1 + 2 * np.sum(autos, axis=0)     # Integrated auto-correlation
    ess = niter / tau                       # Effective sample size
    return lags, autos, tau, ess
