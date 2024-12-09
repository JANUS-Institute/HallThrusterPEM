""" `gen_data.py`

Script to be used with `train_hpc.sh` for generating compression (SVD) data and test set data for training a surrogate.

Call as:

`python gen_data.py <config_file> [--output_dir <output_dir>] [--rank <rank>] [--energy_tol <energy_tol>]
                                  [--compression_samples <compression_samples>] [--test_samples <test_samples>]
                                  [--executor <executor>] [--max_workers <max_workers>]`

Arguments:

- `config_file` - the path to the `amisc` YAML configuration file with the model and input/output variable information.
- `output_dir` - the directory to save all test set and compression data. Defaults to the same path as the config file.
                 If not specified as an 'amisc_{timestamp}' directory, a new directory will be created.
- `rank` - the rank of the SVD compression. Defaults to None, which will defer to `energy_tol`.
- `energy_tol` - the energy tolerance for the SVD compression. Defaults to 0.95.
- `compression_samples` - the number of samples to use for generating the SVD compression data. Defaults to 500.
- `test_samples` - the number of samples to use for generating the test set data. Defaults to 500.
- `executor` - the parallel executor for training surrogate. Options are `thread` or `process`. Default (`process`).
- `max_workers` - the maximum number of workers to use for parallel processing. Defaults to using max number of
                  available CPUs.

Note that a total of `compression_samples` + `test_samples` samples will be generated, which will
run the true underlying models/solvers that many times -- so set accordingly and be prepared for a long runtime.

!!! Note
    New compression data and test set data should be generated anytime _anything_ changes about
    the model or the model inputs. This script should be called before `fit_surr.py`.

Includes:

- `gen_compression_data()` - generate the compression maps for field quantities (only SVD supported).
- `gen_test_set()` - generate a test set for evaluating surrogate performance.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from amisc.typing import COORDS_STR_ID
from uqtils import ax_default
from amisc import YamlLoader, System, to_model_dataset

parser = argparse.ArgumentParser(description=
                                 'Generate compression (SVD) data and test set data for training a surrogate.')
parser.add_argument('config_file', type=str,
                    help='the path to the `amisc` YAML config file with model and input/output variable information.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='the directory to save the generated SVD data and test set data. Defaults to same '
                         'directory as <config_file>.')
parser.add_argument('--rank', type=int, default=None,
                    help='the rank of the SVD compression. Defaults to None, which will defer to `energy_tol`.')
parser.add_argument('--energy_tol', type=float, default=0.95,
                    help='the energy tolerance for the SVD compression. Defaults to 0.95.')
parser.add_argument('--compression_samples', type=int, default=500,
                    help='the number of samples to use for generating the SVD compression data. Defaults to 500.')
parser.add_argument('--test_samples', type=int, default=500,
                    help='the number of samples to use for generating the test set data. Defaults to 500.')
parser.add_argument('--executor', type=str, default='process',
                    help='the parallel executor for training the surrogate. Options are `thread` or `process`. '
                         'Default (`process`).')
parser.add_argument('--max_workers', type=int, default=None,
                    help='the maximum number of workers to use for parallel processing. Defaults to using max'
                         'number of available CPUs.')

args, _ = parser.parse_known_args()


def _extract_grid_coords(field_var: str, model_dir: str | Path):
    """Should have a better way of getting field quantity coordinates directly from the models, which would probably
    require wrapper functions returning the coords -- but this would require some changes in `System.predict` in
    the `amisc` package. See https://github.com/eckelsjd/amisc/issues/31.

    UPDATE: If the wrapper model returns "{{ var }}_coords" in the output, then this will be used as the grid
    coordinates with no need to hard code. Otherwise, will default to this hard code check here.

    For now, just hard code how to extract grid coordinates for each specific field quantity (just `u_ion` and `j_ion`).
    """
    match field_var:
        case 'u_ion':
            files = [f for f in os.listdir(Path(model_dir) / 'Thruster') if f.endswith('.json')]
            with open(Path(model_dir) / 'Thruster' / files[0], 'r') as fd:
                # Assumes hallthruster.jl model output, and assumes the grid is the same for all outputs
                data = json.load(fd)
                coords = np.atleast_1d(data['outputs']['average'][0]['z'])  # axial grid
        case 'j_ion':
            # See hallmd.models.plume.current_density() for how the grid is generated (0 to 90 deg in 1 deg increments)
            coords = np.linspace(0, np.pi/2, 91)
        case other:
            raise ValueError(f"Field quantity '{other}' not recognized. Please add some logic for how to extract "
                             f"the grid coordinates for this field quantity. Or even better, add the variable "
                             f"'{other}_coords' to the model output.")
    return coords


def _object_to_numeric(array: np.ndarray):
    """Helper to convert object arrays of field quantities into a single numeric array by stacking on the first axis.
    Will only work assuming each field quantity has the same shape (which should be true for compression data).
    """
    if np.issubdtype(array.dtype, np.object_):
        return np.concatenate([arr[np.newaxis, ...] for arr in array], axis=0)
    else:
        return array


def _outlier_indices(outputs: dict, iqr_factor: float = 1.5, discard_outliers: bool = True):
    """Compute outliers based on the interquartile range (IQR) method. Outliers are defined as values that are less than
    `Q1 - 1.5 * IQR` or greater than `Q3 + 1.5 * IQR`, where Q1 and Q3 are the 25th and 75th percentiles, respectively.

    :param outputs: dictionary of output arrays, where each array has shape `(num_samples, ...)`.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :returns outlier_idx: boolean array of shape `(num_samples,)` indicating outliers.
    """
    num_samples = next(iter(outputs.values())).shape[0]
    outlier_idx = np.full(num_samples, False)
    nan_idx = np.full(num_samples, False)
    for var, arr in outputs.items():
        if COORDS_STR_ID in str(var):
            continue

        try:
            arr = _object_to_numeric(arr)
        except Exception:
            # Might have object arrays of different shapes, so instead estimate outliers for each sample individually
            if np.issubdtype(arr.dtype, np.object_):
                for i, field_qty in enumerate(arr):
                    nan_idx[i] |= np.any(np.isnan(field_qty))

                    if discard_outliers:
                        p25 = np.percentile(field_qty, 25)
                        p75 = np.percentile(field_qty, 75)
                        iqr = p75 - p25
                        outlier_idx[i] |= np.any((field_qty < p25 - iqr_factor * iqr) |
                                                 (field_qty > p75 + iqr_factor * iqr))
        else:
            # Directly use scalar numeric arrays (or obj arrays that were successfully converted)
            nan_idx |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))

            if discard_outliers:
                p25 = np.percentile(arr, 25, axis=0)
                p75 = np.percentile(arr, 75, axis=0)
                iqr = p75 - p25
                outlier_idx |= np.any((arr < p25 - iqr_factor * iqr) | (arr > p75 + iqr_factor * iqr),
                                      axis=tuple(range(1, arr.ndim)))

    return nan_idx, outlier_idx


def output_bad_sims(samples, outputs, folder, nan_idx, outlier_idx):
    bad_idx = nan_idx | outlier_idx
    num_samples = bad_idx.size

    if np.any(bad_idx):
        num_nan = np.sum(nan_idx)
        num_out = np.sum(outlier_idx)
        system.logger.warning(
            f'Filtered out {np.sum(bad_idx)}/{num_samples} bad samples from the {folder} data '
            f'({num_nan} nan and {num_out} outliers).'
        )

        if num_nan > 0:
            with open(Path(system.root_dir) / folder / 'nans.pkl', 'wb') as fd:
                pickle.dump({'inputs': {str(var): arr[nan_idx, ...] for var, arr in samples.items()},
                             'outputs': {str(var): arr[nan_idx, ...] for var, arr in outputs.items()}}, fd)

        if num_out > 0:
            with open(Path(system.root_dir) / folder / 'outliers.pkl', 'wb') as fd:
                pickle.dump({'inputs': {str(var): arr[outlier_idx, ...] for var, arr in samples.items()},
                             'outputs': {str(var): arr[outlier_idx, ...] for var, arr in outputs.items()}}, fd)


def gen_compression_data(system: System, num_samples: int, rank: int, energy_tol: float, executor: Executor, verbose: bool = False, discard_outliers: bool = True):
    """Compute compression maps for field quantities (only SVD supported).

    Will create a `compression` directory in the `system.root_dir` and save the compression data there.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param num_samples: the number of samples to use for generating the SVD compression data.
    :param rank: the rank of the SVD compression. Defaults to None, which will defer to `energy_tol`.
    :param energy_tol: the energy tolerance for the SVD compression. Defaults to 0.95.
    :param executor: the parallel executor for training the surrogate (i.e. a `concurrent.futures.Executor` instance)
    """
    system.logger.info(f'Generating compression data for {system.name} -- {num_samples} samples...')
    os.mkdir(Path(system.root_dir) / 'compression')
    samples_operating = {v.name: v.normalize(v.sample_domain(num_samples)) for v in system.inputs()
                         if v.category == 'operating'}  # full domain for operating conditions (pdf otherwise)
    samples_params = {v.name: v.normalize(v.sample(num_samples)) for v in system.inputs() if v.category != 'operating'}
    samples = dict(**samples_operating, **samples_params)
    outputs = system.predict(samples, use_model='best', model_dir=Path(system.root_dir) / 'compression',
                             executor=executor, verbose=verbose)

    # Filter bad samples and outliers (by interquartile range)
    nan_idx, outlier_idx = _outlier_indices(outputs, discard_outliers=discard_outliers)
    bad_idx = nan_idx | outlier_idx
    samples, coords = to_model_dataset(samples, system.inputs())
    samples.update(coords)

    with open(Path(system.root_dir) / 'compression' / 'compression.pkl', 'wb') as fd:
        xt = {str(var): arr[~bad_idx, ...] for var, arr in samples.items()}
        yt = {str(var): arr[~bad_idx, ...] for var, arr in outputs.items()}
        pickle.dump({'compression': (xt, yt)}, fd)

    for var in system.outputs():
        if var.compression is not None:
            # Try to get coords from model output; otherwise "hard code" it from output files
            if (coords := outputs.get(f'{var}{COORDS_STR_ID}')) is not None:
                var.compression.coords = coords[0]  # assume all coords are the same for compression data
            else:
                var.compression.coords = _extract_grid_coords(var.name, Path(system.root_dir) / 'compression')

            match var.compression.method.lower():
                case 'svd':
                    data_matrix = {field: var.normalize(_object_to_numeric(outputs[field][~bad_idx, ...])) for field in
                                   var.compression.fields}
                    var.compression.compute_map(data_matrix, rank=rank, energy_tol=energy_tol)
                case other:
                    raise ValueError(f"Compression method '{other}' not supported.")

    system.save_to_file(f'{system.name}_compression.yml', Path(system.root_dir) / 'compression')

    # Write bad sims to file
    output_bad_sims(samples, outputs, 'compression', nan_idx, outlier_idx)

    # Plot and save some results
    for var in system.outputs():
        if var.compression is not None:
            u, s, vt = np.linalg.svd(var.compression.data_matrix)
            energy_frac = np.cumsum(s ** 2 / np.sum(s ** 2))

            fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
            ax.plot(energy_frac, '.k')
            ax.plot(energy_frac[:var.compression.rank], 'or', label=f'Rank={var.compression.rank}')
            ax.axhline(y=var.compression.energy_tol, color='red', linestyle='--', linewidth=1, label='Energy tol')
            ax.set_yscale('log')
            ax_default(ax, 'Singular value index', 'Cumulative energy fraction', legend=True)
            fig.savefig(Path(system.root_dir) / 'compression' / f'{var.name}_svd.png', dpi=300, format='png')

            # Special plots for specific field quantities
            if var.name in ['u_ion', 'j_ion']:
                coords = var.compression.coords
                A = _object_to_numeric(outputs[var.name][~bad_idx, ...])
                lb = np.percentile(A, 5, axis=0)
                mid = np.percentile(A, 50, axis=0)
                ub = np.percentile(A, 95, axis=0)

                fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
                ax.plot(coords, mid, '-k')
                ax.fill_between(coords, lb, ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
                if var.name == 'j_ion':
                    ax.set_yscale('log')
                    xlabel = 'Angle from thruster centerline (rad)'
                else:
                    xlabel = 'Axial location (m)'
                ax_default(ax, xlabel, var.get_tex(units=True, symbol=False), legend=False)
                fig.savefig(Path(system.root_dir) / 'compression' / f'{var.name}_range.png', dpi=300, format='png')


def gen_test_set(system: System, num_samples: int, executor: Executor, verbose: bool = False, discard_outliers: bool = True):
    """Generate a test set of high-fidelity model solves.

    Will create a `test_set` directory in the `system.root_dir` and save the test set data there.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param num_samples: the number of samples to use for generating the test set data.
    :param executor: the parallel executor for training the surrogate (i.e. a `concurrent.futures.Executor` instance)
    """
    system.logger.info(f'Generating test set data for {system.name} -- {num_samples} samples...')
    os.mkdir(Path(system.root_dir) / 'test_set')
    samples_operating = {v.name: v.normalize(v.sample_domain(num_samples)) for v in system.inputs()
                         if v.category == 'operating'}  # full domain for operating conditions (pdf otherwise)
    samples_params = {v.name: v.normalize(v.sample(num_samples)) for v in system.inputs() if v.category != 'operating'}
    samples = dict(**samples_operating, **samples_params)
    outputs = system.predict(samples, use_model='best', model_dir=Path(system.root_dir) / 'test_set',
                             executor=executor, verbose=verbose)

    # Filter bad samples and outliers (by interquartile range)
    nan_idx, outlier_idx = _outlier_indices(outputs, discard_outliers=discard_outliers)
    bad_idx = nan_idx | outlier_idx
    samples, coords = to_model_dataset(samples, system.inputs())
    samples.update(coords)

    with open(Path(system.root_dir) / 'test_set' / 'test_set.pkl', 'wb') as fd:
        xt = {str(var): arr[~bad_idx, ...] for var, arr in samples.items()}
        yt = {str(var): arr[~bad_idx, ...] for var, arr in outputs.items()}
        pickle.dump({'test_set': (xt, yt)}, fd)

    # Write bad samples to file
    output_bad_sims(samples, outputs, 'test_set', nan_idx, outlier_idx)


if __name__ == '__main__':
    system = YamlLoader.load(args.config_file)
    system.root_dir = args.output_dir or Path(args.config_file).parent
    system.set_logger(stdout=True)

    if Path(args.config_file).name not in os.listdir(system.root_dir):
        shutil.copy(args.config_file, system.root_dir)

    match args.executor.lower():
        case 'thread':
            pool_executor = ThreadPoolExecutor
        case 'process':
            pool_executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    with pool_executor(max_workers=args.max_workers) as executor:
        gen_compression_data(system, args.compression_samples, args.rank, args.energy_tol, executor, verbose=True, discard_outliers=False)
        gen_test_set(system, args.test_samples, executor, verbose=True, discard_outliers=False)
        system.logger.info('Data generation complete.')
