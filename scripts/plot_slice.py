""" `plot_slice.py`

Script to be used with `slice_hpc.sh` for plotting 1d slices of the PEM surrogates against the model.

Usage: python plot_slice.py [<config_file>] [--inputs <inputs>] [--outputs <outputs>] [--num_steps <num_steps>]
                                            [--show_model <show_model>] [--model_dir <model_dir>]
                                            [--executor <executor>] [--random_walk] [--max_workers <max_workers>]

Arguments:

- `config_file` - path to an `amisc` YAML surrogate save file. Defaults to searching for the most recent `amisc_*`
                  directory in the path specified by `save_file` (which defaults to the current directory if not
                  specified).
- `inputs` - list of input variables for the slice. Defaults to first 3 inputs.
- `outputs` - list of output variables for the slice. Defaults to first 3 outputs.
- `num_steps` - number of steps for the slice. Defaults to 15 for each input variable.
- `show_model` - list of model fidelities to show. Defaults to `best`.
- `model_dir` - directory to save model outputs. Defaults to the surrogate's root directory, or cwd otherwise.
- `executor` - parallel executor for running the model. Options are `thread` or `process`. Defaults to `process`.
- `random_walk` - flag to perform a random walk. Defaults to `True`. Otherwise, the slice will be a 1d linear grid.
- `max_workers` - maximum number of workers for parallel processing. Defaults to all available CPUs.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import sys
import os
import argparse

import matplotlib.pyplot as plt

from amisc import System


parser = argparse.ArgumentParser(description='Plot 1d slices of the surrogate against the model.')
parser.add_argument('config_file', type=str, nargs='?', default='.', help='Path to the amisc YAML surrogate save file.')
parser.add_argument('--inputs', type=str, nargs='+', default=None, help='List of input variables for the slice.')
parser.add_argument('--outputs', type=str, nargs='+', default=None, help='List of output variables for the slice.')
parser.add_argument('--num_steps', type=int, default=15, help='Number of steps for the slice.')
parser.add_argument('--show_model', type=str, nargs='+', default=['best'], help='List of model fidelities to show.')
parser.add_argument('--model_dir', type=str, default=None, help='Directory to save model outputs.')
parser.add_argument('--executor', type=str, default='process', help='Parallel executor for running the model.')
parser.add_argument('--random_walk', action='store_true', default=True, help='Flag to perform a random walk.')
parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers for parallel processing.')

args, _ = parser.parse_known_args()


def _search_for_config_file(config_file: str | Path):
    """Helper function to find an `amisc` save/config YAML file. If a directory is passed, it will search for the most
    recent timestamp directory and the most recent iteration within that directory.
    """
    # Search for the most recent timestamp
    if Path(config_file).is_dir():
        output_dir = config_file
        if output_dir.startswith('amisc_'):
            pass
        else:
            most_recent = None
            timestamp = 'amisc_2023-01-01T00:00:00'
            for f in os.listdir(Path(output_dir)):
                if (Path(output_dir) / f).is_dir() and f.startswith('amisc_') and f > timestamp:
                    timestamp = f
                    most_recent = f
            if most_recent is not None:
                output_dir = Path(output_dir) / most_recent
            else:
                raise ValueError(f'No amisc timestamp directory found in {output_dir}')

        # Find the last iteration from surrogate training
        last_iteration = -1
        iter_name = None
        for f in os.listdir(Path(output_dir) / 'surrogates'):
            if '_iter' in f:
                iteration = int(f.split('_iter')[-1])
                if iteration > last_iteration:
                    last_iteration = iteration
                    iter_name = f

        if iter_name is not None:
            config_file = Path(output_dir) / 'surrogates' / iter_name / f'{iter_name}.yml'

    if not Path(config_file).exists():
        raise FileNotFoundError(f'No such file or directory: {config_file}')
    else:
        return config_file


if __name__ == '__main__':
    """Plot and save 1d slices of the surrogate against the model for the given PEM configuration file. If the 
    surrogate has no root directory, then outputs will be written to the current directory.
    """
    config_file = _search_for_config_file(args.config_file)  # Will default to a search in current directory

    system = System.load_from_file(config_file)

    if args.model_dir is None:
        model_dir = '.' if system.root_dir is None else system.root_dir
    else:
        model_dir = args.model_dir

    match args.executor:
        case 'thread':
            pool_executor = ThreadPoolExecutor
        case 'process':
            pool_executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    with pool_executor(max_workers=args.max_workers) as executor:
        nominal = {str(var): var.sample_domain((1,)) for var in system.inputs()}  # Random nominal test point
        slice_kwargs = dict(inputs=args.inputs, outputs=args.outputs, num_steps=args.num_steps,
                            show_model=args.show_model, model_dir=model_dir, executor=executor,
                            random_walk=args.random_walk, nominal=nominal)

        fig, ax = system.plot_slice(**slice_kwargs)

    plt.show(block=False)
