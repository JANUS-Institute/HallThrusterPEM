""" `plot_slice.py`

Script to be used with `train.sh` for plotting 1d slices of the PEM surrogates against the model.

USAGE: python plot_slice.py <config_file> [OPTIONS]

REQUIRED:
<config_file>
        the path to the `amisc` YAML configuration file with the model and input/output variable information.

OPTIONS:
-o, --output-dir
        directory to save model outputs. Defaults to the surrogate's root directory, or cwd otherwise.
-e, --executor=thread
        parallel executor for running the model. Options are `thread` or `process`. Defaults to `thread`.
-w, --max-workers
        maximum number of workers for parallel processing. Defaults to all available CPUs.
-s, --search
        flag to search for the most recent amisc timestamp directory and the most recent iteration within that
        directory. If <config_file> is a file, then the search will start from the file's parent directory. Defaults
        to False. Typically, should only let this be set by `train.sh`, since the most recent timestamp directory will
        have just been created. Otherwise, the <config_file> should contain fully-trained surrogate save data.
-I, --inputs
        list of input variables for the slice. Defaults to first 3 inputs.
-O, --outputs
        list of output variables for the slice. Defaults to first 3 outputs.
-N, --num-steps=15
        number of steps for the slice. Defaults to 15 for each input variable.
-M, --show-model
        list of model fidelities to show. Defaults to `best`.
-R, --random-walk
        flag to perform a random walk. Defaults to `True`. Otherwise, the slice will be a 1d linear grid.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import os
import argparse

from amisc import System


parser = argparse.ArgumentParser(description='Plot 1d slices of the surrogate against the model.')
parser.add_argument('config_file', type=str, help='Path to the amisc YAML surrogate save file.')
parser.add_argument('-o', '--output-dir', type=str, default=None, help='Directory to save model outputs.')
parser.add_argument('-e', '--executor', type=str, default='thread', help='Parallel executor for running the model.')
parser.add_argument('-w', '--max-workers', type=int, default=None, help='Maximum number of workers for parallel processing.')
parser.add_argument('-s', '--search', action='store_true', default=False,
                    help='flag to search for the most recent amisc timestamp directory.')
parser.add_argument('-I', '--inputs', type=str, nargs='+', default=None, help='List of input variables for the slice.')
parser.add_argument('-O', '--outputs', type=str, nargs='+', default=None, help='List of output variables for the slice.')
parser.add_argument('-N', '--num-steps', type=int, default=15, help='Number of steps for the slice.')
parser.add_argument('-M', '--show-model', type=str, nargs='+', default=['best'], help='List of model fidelities to show.')
parser.add_argument('-R', '--random-walk', action='store_true', default=True, help='Flag to perform a random walk.')

args, _ = parser.parse_known_args()


def _search_for_config_file(search_dir: str | Path):
    """Helper function to find an `amisc` save/config YAML file. If a directory is passed, it will search for the most
    recent timestamp directory and the most recent iteration within that directory. If a file is passed, the search
    will be performed in the file's parent directory.

    The search is not recursive, so it will only look in the top-level directory.
    """
    if not Path(search_dir).is_dir():
        search_dir = Path(search_dir).parent

    # Search for the most recent amisc timestamp
    if not Path(search_dir).name.startswith('amisc_'):
        most_recent = None
        timestamp = 'amisc_2023-01-01T00:00:00'
        for f in os.listdir(Path(search_dir)):
            if (Path(search_dir) / f).is_dir() and f.startswith('amisc_') and f > timestamp:
                timestamp = f
                most_recent = f
        if most_recent is not None:
            search_dir = Path(search_dir) / most_recent
        else:
            raise ValueError(f'No amisc timestamp directory found in {search_dir}')

    # Find the last iteration from surrogate training
    last_iteration = -1
    iter_name = None
    for f in os.listdir(Path(search_dir) / 'surrogates'):
        if '_iter' in f:
            iteration = int(f.split('_iter')[-1])
            if iteration > last_iteration:
                last_iteration = iteration
                iter_name = f

    if iter_name is not None:
        return Path(search_dir) / 'surrogates' / iter_name / f'{iter_name}.yml'
    else:
        raise ValueError(f'No surrogate iteration found in {search_dir}')


if __name__ == '__main__':
    """Plot and save 1d slices of the surrogate against the model for the given PEM configuration file. If the 
    surrogate has no root directory, then outputs will be written to the current directory.
    """
    config_file = args.config_file
    if args.search:
        config_file = _search_for_config_file(config_file)  # Will search file's parent directory for amisc timestamp

    system = System.load_from_file(config_file)

    if args.output_dir is None:
        save_dir = '.' if system.root_dir is None else system.root_dir
    else:
        save_dir = args.output_dir

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
                            show_model=args.show_model, save_dir=save_dir, executor=executor,
                            random_walk=args.random_walk, nominal=nominal)

        fig, ax = system.plot_slice(**slice_kwargs)
