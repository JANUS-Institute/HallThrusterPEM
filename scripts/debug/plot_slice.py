"""Debugging script to be used with `plot_slice.sh` to test submitting small SLURM jobs."""
from pathlib import Path
import sys
import os


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
    print(str(base_dir.resolve()))
