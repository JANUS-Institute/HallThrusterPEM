""" `fit_surr.py`

Script to be used with `train.sh` for building a multidisciplinary surrogate.

USAGE: `python fit_surr.py <config_file> [OPTIONS]

REQUIRED:
<config_file>
        the path to the `amisc` YAML configuration file with the model and input/output variable information.

OPTIONS:
-o, --output-dir
        the directory to save all surrogate data. Defaults to the same path as the config file.
        If not specified as an 'amisc_{timestamp}' directory, a new directory will be created.
-e, --executor=thread
        the parallel executor for training surrogate. Options are `thread` or `process`. Default to `thread`.
-w, --fit-cpus
        the maximum number of workers to use for parallel processing. Defaults to max available CPUs.
-d, --discard-outliers
        whether to discard outliers from the test set data. Defaults to False.
-s, --search
        whether to search for the most recent compression save file in the output directory. Defaults to False.
        Typically, should only let this be set by `train.sh`, since the compression data will be generated
        immediately before training the surrogate. If you are calling `fit_surr.py` on your own, leave this
        as False, but your <config_file> must have all the data it needs to train the surrogate.
-r, --runtime-hr=3
        the runtime in hours for training the surrogate. Defaults to 3 hours. Will run until completion of
        last iteration past this runtime, which may end up being longer.
-i, --max-iter=200
        the maximum number of iterations to run the surrogate training. Defaults to 200.
-m, --max-tol=1e-3
        the maximum tolerance for the surrogate training. Defaults to 1e-3.
-T, --targets
        the target output variables to train the surrogate on. Defaults to all output variables.
-f, --fidelity=multi
        the fidelity setting for surrogate training. Options are `multi`, `single`, or `both`. Defaults to `multi`.
-n, --save-interval=10
        the interval to save the surrogate during training. Defaults to 10.
-p, --pdf
        whether to use variable PDF weighting during training. Defaults to False.
-C, --cache-interval=0
        the interval to cache component training data during training. Defaults to 0 (no caching).

!!! Note
    The compression and test set data should be generated **first** by running `gen_data.py`.

INCLUDES:
- `train_surrogate()` - train a surrogate from a PEM configuration file.
"""
import argparse
import copy
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default
from amisc import System


parser = argparse.ArgumentParser(description='Train a surrogate from a PEM configuration file.')
parser.add_argument('config_file', type=str,
                    help='the path to the `amisc` YAML config file with model and input/output variable information.')
parser.add_argument('-o', '--output-dir', type=str, default=None,
                    help='the directory to save the generated surrogate data. Defaults to same '
                         'directory as <config_file>.')
parser.add_argument('-e', '--executor', type=str, default='thread', choices=['thread', 'process'],
                    help='the parallel executor for training the surrogate. Options are `thread` or `process`. '
                         'Default to `thread`.')
parser.add_argument('-w', '--fit-cpus', type=int, default=None,
                    help='the maximum number of workers to use for parallel processing. Defaults to using max'
                         'number of available CPUs.')
parser.add_argument('-d', '--discard-outliers', action='store_true', default=False,
                    help='whether to discard outliers from the test set data. Defaults to False.')
parser.add_argument('-s', '--search', action='store_true', default=False,
                    help='whether to search for the most recent compression save file in the output directory.')
parser.add_argument('-r', '--runtime-hr', type=float, default=3.0,
                    help='the maximum runtime in hours for training the surrogate. Defaults to 3 hours.')
parser.add_argument('-i', '--max-iter', type=int, default=200,
                    help='the maximum number of iterations to run the surrogate training. Defaults to 200.')
parser.add_argument('-m', '--max-tol', type=float, default=1e-3,
                    help='the maximum tolerance for the surrogate training. Defaults to 1e-3.')
parser.add_argument('-T', '--targets', type=str, nargs='+', default=None,
                    help='the target output variables to train the surrogate on. Defaults to all output variables.')
parser.add_argument('-f', '--fidelity', type=str, default='multi', choices=['multi', 'single', 'both'],
                    help='the fidelity setting for surrogate training. Options are `multi`, `single`, or `both`.')
parser.add_argument('-n', '--save-interval', type=int, default=10,
                    help='the interval to save the surrogate during training. Defaults to 10.')
parser.add_argument('-p', '--pdf', action='store_true', default=False,
                    help='whether to use variable PDF weighting during training. Defaults to False.')
parser.add_argument('-C', '--cache-interval', type=int, default=0,
                    help='the interval to cache component training data during training. Defaults to 0 (no caching).')

args, _ = parser.parse_known_args()


def train_surrogate(system: System, fidelity: Literal['multi', 'single', 'both'] = 'multi', **fit_kwargs):
    """Train an `amisc.System` surrogate.

    :param system: the `amisc.System` object to train the surrogate on.
    :param fidelity: whether to train a multi-fidelity surrogate, a single-fidelity, or both. Default is multi-fidelity.
                     For multi-fidelity, the config file must have the model_fidelity indices set for each component.
                     For single-fidelity, the model_fidelity indices will be set to empty for each component --
                     the component models should handle this case internally.
    :param fit_kwargs: additional keyword arguments for the surrogate training (passed to `System.fit()`).
    """
    fit_kwargs = dict(num_refine=1000, estimate_bounds=True, update_bounds=True, plot_interval=5, **fit_kwargs)
    targets = fit_kwargs.get('targets', None)
    base_dir = system.root_dir

    if fidelity in ['multi', 'both']:
        system.fit(**fit_kwargs)
        system.plot_allocation()

        mf_cost_alloc, mf_model_cost, mf_overhead_cost, mf_model_evals = system.get_allocation()
        mf_train_history = copy.deepcopy(system.train_history)

        idx = np.where(mf_model_evals != 0)
        system.logger.info(f'Minimum model evaluations per iteration: {np.min(mf_model_evals[idx]):.2f}')
        system.logger.info(f'Average model evaluations per iteration: {np.mean(mf_model_evals[idx]):.2f}')
        system.logger.info(f'Maximum model evaluations per iteration: {np.max(mf_model_evals[idx]):.2f}')

        targets = targets or list(mf_train_history[-1]['test_error'].keys())
        num_plot = min(len(targets), 3)
        mf_test = np.full((len(mf_train_history), num_plot), np.nan)
        for j, mf_res in enumerate(mf_train_history):
            for i, var in enumerate(targets[:num_plot]):
                if (perf := mf_res.get('test_error')) is not None:
                    mf_test[j, i] = perf[var]

        # Get the cost of a single high-fidelity evaluation (for each model separately, and then sum)
        highest_cost = []
        for comp in system.components:
            if comp.name in mf_cost_alloc:
                highest_cost.append(max(system[comp.name].model_costs.values()))
        highest_cost = sum(highest_cost)

    if fidelity in ['single', 'both']:
        # Reset data for single-fidelity training
        system.clear()
        for comp in system.components:
            comp.model_fidelity = ()   # Should use each model's default value for model_fidelity if applicable

        if fidelity == 'both':
            system.root_dir = system.root_dir / 'amisc_single_fidelity'

        system.fit(**fit_kwargs)

        sf_cost_alloc, sf_model_cost, sf_overhead_cost, sf_model_evals = system.get_allocation()
        sf_train_history = copy.deepcopy(system.train_history)

        idx = np.where(sf_model_evals != 0)
        system.logger.info(f'Minimum model evaluations per iteration: {np.min(sf_model_evals[idx]):.2f}')
        system.logger.info(f'Average model evaluations per iteration: {np.mean(sf_model_evals[idx]):.2f}')
        system.logger.info(f'Maximum model evaluations per iteration: {np.max(sf_model_evals[idx]):.2f}')

        targets = targets or list(sf_train_history[-1]['test_error'].keys())
        num_plot = min(len(targets), 3)
        sf_test = np.full((len(sf_train_history), num_plot), np.nan)
        for j, sf_res in enumerate(sf_train_history):
            for i, var in enumerate(targets[:num_plot]):
                if (perf := sf_res.get('test_error')) is not None:
                    sf_test[j, i] = perf[var]

        # Get the cost of a single high-fidelity evaluation (for each model separately, and then sum)
        highest_cost = []
        for comp in system.components:
            if comp.name in sf_cost_alloc:
                highest_cost.append(max(system[comp.name].model_costs.values()))
        highest_cost = sum(highest_cost)

    # Plot QoI L2 error on test set vs. cost
    labels = [system.outputs()[var].get_tex(units=True) for var in targets]
    fig, axs = plt.subplots(1, num_plot, sharey='row', figsize=(3.5 * num_plot, 4), layout='tight', squeeze=False)
    for i in range(num_plot):
        ax = axs[0, i]
        if fidelity in ['multi', 'both']:
            ax.plot(np.cumsum(mf_model_cost) / highest_cost, mf_test[:, i], '-k', label='Multi-fidelity (MF)')
        if fidelity in ['single', 'both']:
            ax.plot(np.cumsum(sf_model_cost) / highest_cost, sf_test[:, i], '--k', label='Single-fidelity (SF)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.set_title(labels[i])
        ylabel = r'Relative error' if i == 0 else ''
        ax_default(ax, r'Cost (number of model evals)', ylabel, legend=i+1 == num_plot)
    fig.savefig(base_dir / 'error_v_cost.pdf', bbox_inches='tight', format='pdf')

    # Plot algorithm overhead
    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    if fidelity in ['multi', 'both']:
        ax.plot(np.cumsum(mf_overhead_cost) / highest_cost, '-k', label='Multi-fidelity (MF)')
    if fidelity in ['single', 'both']:
        ax.plot(np.cumsum(sf_overhead_cost) / highest_cost, '--k', label='Single-fidelity (SF)')
    ax.set_yscale('log')
    ax.grid()
    ax_default(ax, 'Iteration', 'Overhead cost (number of model evals)', legend=True)
    fig.savefig(base_dir / 'overhead.pdf', bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    output_dir = args.output_dir or Path(args.config_file).parent
    system = None

    # Search for an amisc compression save directory generated by `gen_data.py`
    if args.search:
        if not str(output_dir).startswith('amisc_'):
            # Search for the most recent amisc timestamp
            most_recent = None
            timestamp = 'amisc_2023-01-01T00:00:00'
            for f in os.listdir(Path(output_dir)):
                if (Path(output_dir) / f).is_dir() and f.startswith('amisc_') and f > timestamp:
                    timestamp = f
                    most_recent = f

            if most_recent is not None:
                output_dir = Path(output_dir) / most_recent

        # Now try to load from a compression save file
        if str(Path(output_dir).name).startswith('amisc_'):
            if (Path(output_dir) / 'compression').exists():
                compression_files = list((Path(output_dir) / 'compression').glob('*_compression.yml')) # See gen_data.py
                if compression_files:
                    compression_file = compression_files[0]
                    system = System.load_from_file(compression_file)

    # If not searching or couldn't find a compression save file, load from the config file directly
    if system is None:
        system = System.load_from_file(args.config_file, output_dir)

    if Path(args.config_file).name not in os.listdir(system.root_dir):
        shutil.copy(args.config_file, system.root_dir)

    match args.executor.lower():
        case 'thread':
            pool_executor = ThreadPoolExecutor
        case 'process':
            pool_executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    # Load the test set data
    base_dir = system.root_dir
    test_set = pth if (pth := base_dir / 'test_set' / 'test_set.pkl').exists() else None

    if test_set is not None:
        with open(test_set, 'rb') as fd:
            data = pickle.load(fd)
            num_samples = next(iter(data['test_set'][0].values())).shape[0]
            discard_idx = np.full(num_samples, False)

            for var, idx in data.get('nan_idx', {}).items():
                discard_idx |= idx

            if args.discard_outliers:
                for var, idx in data.get('outlier_idx', {}).items():
                    discard_idx |= idx

            test_set = ({k: v[~discard_idx, ...] for k, v in data['test_set'][0].items()},
                        {k: v[~discard_idx, ...] for k, v in data['test_set'][1].items()})

    with pool_executor(max_workers=args.fit_cpus) as executor:
        fit_kwargs = {'runtime_hr': args.runtime_hr, 'max_iter': args.max_iter, 'targets': args.targets,
                      'save_interval': args.save_interval, 'max_tol': args.max_tol, 'test_set': test_set,
                      'executor': executor, 'weight_fcns': None if not args.pdf else 'pdf',
                      'cache_interval': args.cache_interval}
        train_surrogate(system, fidelity=args.fidelity, **fit_kwargs)

    system.logger.info(f'Surrogate training complete. Output directory: {base_dir}')
