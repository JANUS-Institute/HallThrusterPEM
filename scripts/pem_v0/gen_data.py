""" `gen_data.py`

Script to be used with `gen_data.sh` for generating the SVD data and test set for building a surrogate of PEM v0.

!!! Note
    The SVD data for PEM v0 should be generated **first**. Then the SVD data (i.e. `thruster_svd.pkl` and
    `plume_svd.pkl`) can be used to generate a `test_set.pkl`. These data files will all be placed in `models/config`
    by running this script. This should be done anytime _anything_ changes about the model or the model inputs. This
    script must be run **before** `fit_surr.py`.

Includes
--------
- `gen_svd_data()` - generate SVD training data for the thruster and plume models.
- `gen_test_set()` - generate a test set for PEM v0.
"""
import datetime
from datetime import timezone
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default

from hallmd.models.pem import pem_v0
from hallmd.utils import model_config_dir

CONFIG_DIR = model_config_dir()


def gen_svd_data(N=500, r_pct=0.95):
    """Generate data matrices for SVD dimension reduction."""
    # Thruster svd dataset for uion velocity profile
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path('results') / f'svd_{timestamp}'
    os.mkdir(root_dir)
    surr = pem_v0(executor=None, init=False, save_dir=root_dir)
    xt = surr.sample_inputs(N, comp='Thruster', use_pdf=True)
    comp = surr['Thruster']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, use_model='best', model_dir=comp._model_kwargs.get('output_dir'))
    nan_idx = np.any(np.isnan(yt), axis=-1)
    yt = yt[~nan_idx, :]
    A = yt[:, 7:]       # Data matrix, uion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s**2 / np.sum(s**2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1         # Number of singular values to keep
    vtr = vt[:r, :]     # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open(root_dir / 'thruster_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str(root_dir/'thruster_svd.png'), dpi=300, format='png')
    fig, ax = plt.subplots()
    M = vtr.shape[1]            # Number of grid cells
    zg = np.linspace(0, 1, M)   # Normalized grid locations
    lb = np.percentile(A, 5, axis=0)
    mid = np.percentile(A, 50, axis=0)
    ub = np.percentile(A, 95, axis=0)
    ax.plot(zg, mid, '-k')
    ax.fill_between(zg, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax_default(ax, 'Normalized axial location', 'Ion velocity (m/s)', legend=False)
    fig.savefig(str(root_dir/'uion.png'), dpi=300, format='png')

    # Generate SVD data matrix for Plume
    xt = surr.sample_inputs(N, comp='Plume', use_pdf=True)
    comp = surr['Plume']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, use_model='best')
    idx = ~np.isnan(yt[:, 0]) & (np.nanmax(yt, axis=-1) <= 500)    # Remove some outliers above 500 A/m^2
    yt = yt[idx, :]
    A = yt[:, 2:]  # Data matrix, jion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s ** 2 / np.sum(s ** 2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1  # Number of singular values to keep
    vtr = vt[:r, :]  # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open(root_dir / 'plume_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str(root_dir / 'plume_svd.png'), dpi=300, format='png')
    fig, ax = plt.subplots()
    M = vtr.shape[1]  # Number of grid cells
    zg = np.linspace(0, 1, M)  # Normalized grid locations
    lb = np.percentile(A, 5, axis=0)
    mid = np.percentile(A, 50, axis=0)
    ub = np.percentile(A, 95, axis=0)
    ax.plot(zg, mid, '-k')
    ax.fill_between(zg, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax.set_yscale('log')
    ax_default(ax, 'Normalized angle', 'Current density ($A/m^2$)', legend=False)
    fig.savefig(str(root_dir / 'jion.png'), dpi=300, format='png')

    return root_dir


def gen_test_set(N=500):
    """Generate a test set of high-fidelity model solves."""
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
    root_dir = Path('results') / f'test_{timestamp}'
    os.mkdir(root_dir)
    surr = pem_v0(executor=None, init=False, save_dir=root_dir)
    xt = surr.sample_inputs(N, use_pdf=True)     # (N, xdim)
    yt = surr(xt, use_model='best', model_dir=surr['Thruster']._model_kwargs.get('output_dir'))
    nan_idx = np.any(np.isnan(yt), axis=-1)
    xt = xt[~nan_idx, :]
    yt = yt[~nan_idx, :]
    data = {'xt': xt, 'yt': yt}

    with open(Path(root_dir) / 'test_set.pkl', 'wb') as fd:
        pickle.dump(data, fd)

    return root_dir


if __name__ == '__main__':
    # Generate SVD and test set files
    svd_dir = gen_svd_data()
    shutil.copyfile(svd_dir / 'plume_svd.pkl', CONFIG_DIR / 'plume_svd.pkl')
    shutil.copyfile(svd_dir / 'thruster_svd.pkl', CONFIG_DIR / 'thruster_svd.pkl')
    test_dir = gen_test_set()
    shutil.copyfile(test_dir / 'test_set.pkl', CONFIG_DIR / 'test_set.pkl')
