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

PROJECT_ROOT = Path('../..')
TRAINING = False
CONFIG_DIR = model_config_dir()
surr_dir = list((PROJECT_ROOT / 'results' / 'mf_2024-07-05T18.36.17' / 'multi-fidelity').glob('amisc_*'))[0]
SURR = pem_v0(from_file=surr_dir / 'sys' / f'sys_final{"_train" if TRAINING else ""}.pkl')

def gen_svd_data(N=500, r_pct=0.95):
    """Generate data matrices for SVD dimension reduction."""
    # Thruster svd dataset for uion velocity profile
    xt = SURR.sample_inputs(N, comp='Thruster', use_pdf=True)
    comp = SURR['Thruster']
    comp._model_kwargs['compress'] = False
    yt = comp(xt, model_dir=comp._model_kwargs.get('output_dir'))
    nan_idx = np.any(np.isnan(yt), axis=-1)
    yt = yt[~nan_idx, :]
    A = yt[:, 7:]       # Data matrix, uion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s**2 / np.sum(s**2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1         # Number of singular values to keep
    vtr = vt[:r, :]     # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open('thruster_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str('thruster_svd.png'), dpi=300, format='png')
    fig, ax = plt.subplots()
    M = vtr.shape[1]            # Number of grid cells
    zg = np.linspace(0, 1, M)   # Normalized grid locations
    lb = np.percentile(A, 5, axis=0)
    mid = np.percentile(A, 50, axis=0)
    ub = np.percentile(A, 95, axis=0)
    ax.plot(zg, mid, '-k')
    ax.fill_between(zg, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax_default(ax, 'Normalized axial location', 'Ion velocity (m/s)', legend=False)
    fig.savefig(str('uion.png'), dpi=300, format='png')

    # Generate SVD data matrix for Plume
    xt = SURR.sample_inputs(N, comp='Plume', use_pdf=True)
    comp = SURR['Plume']
    comp._model_kwargs['compress'] = False
    yt = comp(xt)
    idx = ~np.isnan(yt[:, 0]) & (np.nanmax(yt, axis=-1) <= 500)    # Remove some outliers above 500 A/m^2
    yt = yt[idx, :]
    A = yt[:, 2:]  # Data matrix, jion (N x M)
    u, s, vt = np.linalg.svd((A - np.mean(A, axis=0)) / np.std(A, axis=0))
    frac = np.cumsum(s ** 2 / np.sum(s ** 2))
    idx = int(np.where(frac >= r_pct)[0][0])
    r = idx + 1  # Number of singular values to keep
    vtr = vt[:r, :]  # (r x M)
    save_dict = {'A': A, 'r_pct': r_pct, 'r': r, 'vtr': vtr}
    with open('plume_svd.pkl', 'wb') as fd:
        pickle.dump(save_dict, fd)

    # Plot and save some results
    fig, ax = plt.subplots()
    ax.plot(s, '.k')
    ax.plot(s[:r], 'or')
    ax.set_yscale('log')
    ax.set_title(f'r = {r}')
    ax_default(ax, 'Index', 'Singular value', legend=False)
    fig.savefig(str('plume_svd.png'), dpi=300, format='png')
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
    fig.savefig(str('jion.png'), dpi=300, format='png')

    # Reconstruct the data using the first r singular values/vectors
    A_reconstructed = np.dot(u[:, :r], np.dot(np.diag(s[:r]), vt[:r, :]))

    # Plot original vs. reconstructed data for comparison
    fig, ax = plt.subplots()
    ax.plot(A, label='Original Data')
    ax.plot(A_reconstructed, label='Reconstructed Data')
    ax.legend()
    ax.set_title('Comparison of Original and Reconstructed Data')
    ax_default(ax, 'Data Index', 'Data Value', legend=True)
    fig.savefig('reconstruction_comparison.png', dpi=300, format='png')


if __name__ == '__main__':
    # Generate SVD and test set files
    gen_svd_data()
