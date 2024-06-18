""" `plot_slice.py`

Script to be used with `plot_slice.sh` for plotting 1d slices of the PEM v0 surrogate against the model.

Usage: python plot_slice.py <root_dir>

- `root_dir` - the base directory of the mf_timestamp surrogate save directory in the `results` folder. Defaults to 'recent',
               which will search for the most recent timestamp.
"""
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt

from amisc.system import SystemSurrogate


def slice_1d(surr_dir):
    # Load the trained system surrogate
    surr = SystemSurrogate.load_from_file(surr_dir / 'sys' / 'sys_final.pkl')

    # 1d slice test set(s) for plotting
    N = 25
    slice_idx = ['c_w', 'f_n', 'vAN1', 'vAN2', 'vAN3', 'vAN4', 'z0', 'p0']
    qoi_idx = ['I_D', 'T', 'uion0', 'uion1']
    nominal = {str(var): var.sample_domain((1,)) for var in surr.exo_vars}  # Random nominal test point
    fig, ax = surr.plot_slice(slice_idx, qoi_idx, show_model=['best', 'worst'], show_surr=True, N=N,
                              random_walk=True, nominal=nominal, model_dir=surr_dir)
    plt.show()


if __name__ == '__main__':
    mf_timestamp = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1].startswith('mf_') else "recent"

    # Search for the most recent timestamp
    if mf_timestamp == 'recent':
        files = os.listdir(Path('results'))
        mf_timestamp = 'mf_2023-01-01T00:00:00'
        for f in files:
            if (Path('results')/f).is_dir() and f.startswith('mf_') and f > mf_timestamp:
                mf_timestamp = f

    base_dir = list((Path('results')/mf_timestamp/'multi-fidelity').glob('amisc_*'))[0]
    slice_1d(base_dir)
