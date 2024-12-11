""" `gen_data.py`

Script to be used with `train_hpc.sh` for generating compression (SVD) data and test set data for training a surrogate.

Call as:

`python gen_data.py <config_file> [--output_dir <output_dir>] [--rank <rank>] [--energy_tol <energy_tol>]
                                  [--reconstruction_tol <reconstruction_tol>] [--discard_outliers] [--iqr_factor <iqr_factor>]
                                  [--compression_samples <compression_samples>] [--test_samples <test_samples>]
                                  [--executor <executor>] [--max_workers <max_workers>]`

Arguments:

- `config_file` - the path to the `amisc` YAML configuration file with the model and input/output variable information.
- `output_dir` - the directory to save all test set and compression data. Defaults to the same path as the config file.
                 If not specified as an 'amisc_{timestamp}' directory, a new directory will be created.
- `rank` - the rank of the SVD compression. Defaults to None, which will defer to `energy_tol` or `reconstruction_tol`.
- `energy_tol` - the energy tolerance for the SVD compression. Defaults to None, which will defer to `reconstruction_tol`.
- `reconstruction_tol` - the reconstruction error tolerance for the SVD compression. Defaults to 0.05.
- `compression_samples` - the number of samples to use for generating the SVD compression data. Defaults to 500.
- `discard_outliers` - whether to discard outliers from the compression and test set data. Defaults to False.
- `iqr_factor` - the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
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
from logging import Logger
import pickle

import numpy as np
import matplotlib.pyplot as plt
from amisc.typing import COORDS_STR_ID
from uqtils import ax_default
from amisc import YamlLoader, System, to_model_dataset, VariableList

parser = argparse.ArgumentParser(description=
                                 'Generate compression (SVD) data and test set data for training a surrogate.')
parser.add_argument('config_file', type=str,
                    help='the path to the `amisc` YAML config file with model and input/output variable information.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='the directory to save the generated SVD data and test set data. Defaults to same '
                         'directory as <config_file>.')
parser.add_argument('--rank', type=int, default=None,
                    help='the rank of the SVD compression. Defaults to None, which will defer to `energy_tol`'
                         'or `reconstruction_tol`.')
parser.add_argument('--energy_tol', type=float, default=None,
                    help='the energy tolerance for the SVD compression. Defaults to None, which will defer to '
                         '`reconstruction_tol`.')
parser.add_argument('-reconstruction_tol', type=float, default=0.05,
                    help='the reconstruction tolerance for the SVD compression. Defaults to 0.05.')
parser.add_argument('--compression_samples', type=int, default=500,
                    help='the number of samples to use for generating the SVD compression data. Defaults to 500.')
parser.add_argument('--test_samples', type=int, default=500,
                    help='the number of samples to use for generating the test set data. Defaults to 500.')
parser.add_argument('--discard_outliers', action='store_true', default=False,
                    help='whether to discard outliers from the compression and test set data. Defaults to False.')
parser.add_argument('--iqr_factor', type=float, default=1.5,
                    help='the factor to multiply the IQR by to determine outliers. Defaults to 1.5.')
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


def _filter_outputs(outputs: dict, iqr_factor: float = 1.5, discard_outliers: bool = False, logger: Logger = None):
    """Return indices of outputs to discard (either nan or outliers).

    Compute outliers based on the interquartile range (IQR) method. Outliers are defined as values that are less than
    `Q1 - 1.5 * IQR` or greater than `Q3 + 1.5 * IQR`, where Q1 and Q3 are the 25th and 75th percentiles, respectively.

    :param outputs: dictionary of output arrays, where each array has shape `(num_samples, ...)`.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :param discard_outliers: whether to include any detected outliers in the returned `discard_idx`
    :param logger: a logger for showing warnings if nans or outliers are detected (will print to console by default)
    :returns: `discard_idx` and `outlier_idx` -- a boolean array of shape `(num_samples,)` indicating samples to
              discard, and a `dict` with a boolean array of shape `(num_samples,)` indicating outliers for each output
    """
    warning_func = print if logger is None else logger.warning
    num_samples = next(iter(outputs.values())).shape[0]
    outlier_idx = {}
    nan_idx = np.full(num_samples, False)
    cnt_thresh = 0.75  # Only count a QoI as an outlier if more than 75% of its values are outliers; always true for scalars

    for var, arr in outputs.items():
        if COORDS_STR_ID in str(var):
            continue

        try:
            arr = _object_to_numeric(arr)
        except Exception:
            # Might have object arrays of different shapes, so instead estimate outliers for each sample individually
            if np.issubdtype(arr.dtype, np.object_):
                outlier_idx.setdefault(var, np.full(num_samples, False))
                for i, field_qty in enumerate(arr):
                    nan_idx[i] |= np.any(np.isnan(field_qty))

                    p25 = np.percentile(field_qty, 25)
                    p75 = np.percentile(field_qty, 75)
                    iqr = p75 - p25
                    outlier_idx[var][i] |= np.sum((field_qty < p25 - iqr_factor * iqr) |
                                                  (field_qty > p75 + iqr_factor * iqr)) > int(cnt_thresh * field_qty.size)
        else:
            # Directly use scalar numeric arrays (or obj arrays that were successfully converted)
            nan_idx |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))

            outlier_idx.setdefault(var, np.full(num_samples, False))
            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)
            iqr = p75 - p25
            outlier_idx[var] |= np.sum((arr < p25 - iqr_factor * iqr) | (arr > p75 + iqr_factor * iqr),
                                       axis=tuple(range(1, arr.ndim))) > int(cnt_thresh * np.prod(arr.shape[1:]))

    discard_idx = nan_idx

    all_outliers = np.full_like(nan_idx, False)
    for var, idx in outlier_idx.items():
        all_outliers |= idx

    if np.any(nan_idx):
        warning_func(f'Discarded {np.sum(nan_idx)}/{num_samples} samples with nans.')
    if np.any(all_outliers):
        warning_func(f'Detected {np.sum(all_outliers)}/{num_samples} outliers.')

    if discard_outliers:
        warning_func(f'Discarding outliers...')
        discard_idx |= all_outliers

    return discard_idx, outlier_idx


def _normalize_outputs(outputs: dict, variables: VariableList):
    """Normalize outputs for outlier detection.

    :param outputs: dictionary of output arrays, where each array has shape `(num_samples, ...)`.
    :param variables: list of `Variable` objects for the output quantities
    :returns: dictionary of normalized output arrays
    """
    norm_outputs = {}
    for var in variables:
        if var in outputs:
            norm_outputs[var.name] = var.normalize(_object_to_numeric(outputs[var.name]))

    return norm_outputs


def gen_compression_data(system: System,
                         num_samples: int = 500,
                         rank: int = None,
                         energy_tol: float = None,
                         reconstruction_tol: float = 0.05,
                         iqr_factor: float = 1.5,
                         executor: Executor = None,
                         verbose: bool = False,
                         discard_outliers: bool = False):
    """Compute compression maps for field quantities (only SVD supported).

    Will create a `compression` directory in the `system.root_dir` and save the compression data there.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param num_samples: the number of samples to use for generating the SVD compression data.
    :param rank: the rank of the SVD compression. Defaults to None, which will defer to `energy_tol`.
    :param energy_tol: the energy tolerance for the SVD compression. Defaults to 0.95.
    :param reconstruction_tol: the reconstruction tolerance for the SVD compression. Defaults to 0.05.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :param executor: the parallel executor for training the surrogate (i.e. a `concurrent.futures.Executor` instance)
    :param verbose: whether to print verbose output. Defaults to False.
    :param discard_outliers: whether to discard outliers from the compression data. Defaults to False.
    """
    system.logger.info(f'Generating compression data for {system.name} -- {num_samples} samples...')
    os.mkdir(Path(system.root_dir) / 'compression')
    samples_operating = {v.name: v.normalize(v.sample_domain(num_samples)) for v in system.inputs()
                         if v.category == 'operating'}  # full domain for operating conditions (pdf otherwise)
    samples_params = {v.name: v.normalize(v.sample(num_samples)) for v in system.inputs() if v.category != 'operating'}
    samples = dict(**samples_operating, **samples_params)
    outputs = system.predict(samples, use_model='best', model_dir=Path(system.root_dir) / 'compression',
                             executor=executor, verbose=verbose)

    samples, coords = to_model_dataset(samples, system.inputs())
    samples.update(coords)

    norm_outputs = _normalize_outputs(outputs, system.outputs())

    # Filter bad samples and outliers in norm space (by interquartile range)
    discard_idx, outlier_idx = _filter_outputs(norm_outputs, discard_outliers=discard_outliers, iqr_factor=iqr_factor,
                                               logger=system.logger)

    with open(Path(system.root_dir) / 'compression' / 'compression.pkl', 'wb') as fd:
        dump = {'data': ({str(var): arr[~discard_idx, ...] for var, arr in samples.items()},
                         {str(var): arr[~discard_idx, ...] for var, arr in outputs.items()}),
                'outliers': {}}

        for var, idx in outlier_idx.items():
            if np.any(idx):
                dump['outliers'][str(var)] = ({str(v): arr[idx, ...] for v, arr in samples.items()},
                                              {str(v): arr[idx, ...] for v, arr in outputs.items()})

        pickle.dump(dump, fd)

    for var in system.outputs():
        if var.compression is not None:
            # Try to get coords from model output; otherwise "hard code" it from output files
            if (coords := outputs.get(f'{var}{COORDS_STR_ID}')) is not None:
                var.compression.coords = coords[0]  # assume all coords are the same for compression data
            else:
                var.compression.coords = _extract_grid_coords(var.name, Path(system.root_dir) / 'compression')

            match var.compression.method.lower():
                case 'svd':
                    data_matrix = {field: var.normalize(_object_to_numeric(outputs[field][~discard_idx, ...]))
                                   for field in var.compression.fields}
                    var.compression.compute_map(data_matrix, rank=rank, energy_tol=energy_tol,
                                                reconstruction_tol=reconstruction_tol)
                case other:
                    raise ValueError(f"Compression method '{other}' not supported.")

    system.save_to_file(f'{system.name}_compression.yml', Path(system.root_dir) / 'compression')

    # Plot and save some results
    for var in system.outputs():
        if var.compression is not None:
            u, s, vt = np.linalg.svd(var.compression.data_matrix)
            energy_frac = np.cumsum(s ** 2 / np.sum(s ** 2))

            fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout='tight')
            ax[0].plot(s, '.k')
            ax[0].plot(s[:var.compression.rank], 'or', label=f'Rank={var.compression.rank}')
            ax[0].set_yscale('log')
            ax[0].grid()
            ax_default(ax[0], 'Singular value index', 'Singular value', legend=True)
            ax[1].plot(energy_frac, '.k')
            ax[1].plot(energy_frac[:var.compression.rank], 'or', label=f'Rank={var.compression.rank}')
            ax[1].axhline(y=var.compression.energy_tol, color='red', linestyle='--', linewidth=1, label='Energy tol')
            ax[1].set_yscale('log')
            ax[1].grid()
            ax_default(ax[1], 'Singular value index', 'Cumulative energy fraction', legend=True)
            fig.savefig(Path(system.root_dir) / 'compression' / f'{var.name}_svd.png', dpi=300, format='png')

            # Special plots for specific field quantities
            if var.name in ['u_ion', 'j_ion']:
                coords = var.compression.coords
                A = _object_to_numeric(outputs[var.name][~discard_idx, ...])
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
                ax.grid()
                ax_default(ax, xlabel, var.get_tex(units=True, symbol=False), legend=False)
                fig.savefig(Path(system.root_dir) / 'compression' / f'{var.name}_range.png', dpi=300, format='png')


def gen_test_set(system: System,
                 num_samples: int = 500,
                 executor: Executor = None,
                 verbose: bool = False,
                 discard_outliers: bool = False,
                 iqr_factor: float = 1.5):
    """Generate a test set of high-fidelity model solves.

    Will create a `test_set` directory in the `system.root_dir` and save the test set data there.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param num_samples: the number of samples to use for generating the test set data.
    :param executor: the parallel executor for training the surrogate (i.e. a `concurrent.futures.Executor` instance)
    :param verbose: whether to print verbose output. Defaults to False.
    :param discard_outliers: whether to discard outliers from the test set data. Defaults to False.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    """
    system.logger.info(f'Generating test set data for {system.name} -- {num_samples} samples...')
    os.mkdir(Path(system.root_dir) / 'test_set')
    samples_operating = {v.name: v.normalize(v.sample_domain(num_samples)) for v in system.inputs()
                         if v.category == 'operating'}  # full domain for operating conditions (pdf otherwise)
    samples_params = {v.name: v.normalize(v.sample(num_samples)) for v in system.inputs() if v.category != 'operating'}
    samples = dict(**samples_operating, **samples_params)
    outputs = system.predict(samples, use_model='best', model_dir=Path(system.root_dir) / 'test_set',
                             executor=executor, verbose=verbose)

    samples, coords = to_model_dataset(samples, system.inputs())
    samples.update(coords)

    norm_outputs = _normalize_outputs(outputs, system.outputs())

    # Filter bad samples and outliers (by interquartile range)
    discard_idx, outlier_idx = _filter_outputs(norm_outputs, discard_outliers=discard_outliers, iqr_factor=iqr_factor,
                                               logger=system.logger)

    with open(Path(system.root_dir) / 'test_set' / 'test_set.pkl', 'wb') as fd:
        dump = {'test_set': ({str(var): arr[~discard_idx, ...] for var, arr in samples.items()},
                             {str(var): arr[~discard_idx, ...] for var, arr in outputs.items()}),
                'outliers': {}}

        for var, idx in outlier_idx.items():
            if np.any(idx):
                dump['outliers'][str(var)] = ({str(v): arr[idx, ...] for v, arr in samples.items()},
                                              {str(v): arr[idx, ...] for v, arr in outputs.items()})

        pickle.dump(dump, fd)


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
        gen_compression_data(system, num_samples=args.compression_samples, rank=args.rank, energy_tol=args.energy_tol,
                             reconstruction_tol=args.reconstruction_tol, executor=executor, verbose=True,
                             discard_outliers=args.discard_outliers, iqr_factor=args.iqr_factor)
        gen_test_set(system, num_samples=args.test_samples, executor=executor, verbose=True,
                     discard_outliers=args.discard_outliers, iqr_factor=args.iqr_factor)
        system.logger.info('Data generation complete.')
