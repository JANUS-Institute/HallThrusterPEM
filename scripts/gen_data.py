""" `gen_data.py`

Script to be used with `train.sh` for generating compression (SVD) data and test set data for training a surrogate.
The type of compression (and additional parameters, i.e. rank, tol, etc.) for each variable should be specified in
the config file.

USAGE: `python gen_data.py <config_file> [OPTIONS]

REQUIRED:
<config_file>
        the path to the `amisc` YAML configuration file with the model and input/output variable information.

OPTIONS:
-o, --output-dir
        the directory to save all test set and compression data. Defaults to the same path as the config file.
        If not specified as an 'amisc_{timestamp}' directory, a new directory will be created.
-e, --executor=thread
        the parallel executor for evaluating the models. Options are `thread` or `process`. Defaults to `thread`.
-w, --gen-cpus
        the maximum number of workers to use for parallel processing. Defaults to using max available CPUs.
-d, --discard-outliers
        whether to discard outliers from the compression and test set data. Defaults to False.
-c, --compression-samples=500
        the number of samples to use for generating the SVD compression data. Defaults to 500.
-t, --test-samples=500
        the number of samples to use for generating the test set data. Defaults to 500.
-q, --iqr-factor=1.5
        the factor to multiply the IQR by to determine outliers. Defaults to 1.5.

Note that a total of `compression_samples` + `test_samples` samples will be generated, which will
run the true underlying models/solvers that many times -- so set accordingly and be prepared for a long runtime.

!!! Note
    New compression data and test set data should be generated anytime _anything_ changes about
    the model or the model inputs. This script should be called before `fit_surr.py`.

INCLUDES:
- `generate_data()` - generate the compression or test set data for training a surrogate.
- `process_compression()` - compute the compression maps for field quantities (only SVD supported).
- `plot_compression()` - create plots of the compression data.
- `plot_test_set()` - create plots of the test set data.
- `plot_outliers()` - plot histograms of outliers detected in the output data.
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
parser.add_argument('-o', '--output-dir', type=str, default=None,
                    help='the directory to save the generated SVD data and test set data. Defaults to same '
                         'directory as <config_file>.')
parser.add_argument('-e', '--executor', type=str, default='thread', choices=['thread', 'process'],
                    help='the parallel executor for evaluating the models. Options are `thread` or `process`. '
                         'Default to `thread`.')
parser.add_argument('-w', '--gen-cpus', type=int, default=None,
                    help='the maximum number of workers to use for parallel processing. Defaults to using max'
                         'number of available CPUs.')
parser.add_argument('-d', '--discard-outliers', action='store_true', default=False,
                    help='whether to discard outliers from the compression and test set data. Defaults to False.')
parser.add_argument('-c', '--compression-samples', type=int, default=500,
                    help='the number of samples to use for generating the SVD compression data. Defaults to 500.')
parser.add_argument('-t', '--test-samples', type=int, default=500,
                    help='the number of samples to use for generating the test set data. Defaults to 500.')
parser.add_argument('-q', '--iqr-factor', type=float, default=1.5,
                    help='the factor to multiply the IQR by to determine outliers. Defaults to 1.5.')


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


def _filter_outputs(outputs: dict, iqr_factor: float = 1.5):
    """Return indices of outputs to discard (either nan or outliers).

    Compute outliers based on the interquartile range (IQR) method. Outliers are defined as values that are less than
    `Q1 - 1.5 * IQR` or greater than `Q3 + 1.5 * IQR`, where Q1 and Q3 are the 25th and 75th percentiles, respectively.

    :param outputs: dictionary of output arrays, where each array has shape `(num_samples, ...)`.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :returns: `nan_idx` and `outlier_idx` -- a `dict` with boolean arrays of shape `(num_samples,)` indicating nan
              samples for each output, and a `dict` with a boolean array of shape `(num_samples,)` indicating outliers
              for each output
    """
    num_samples = next(iter(outputs.values())).shape[0]
    outlier_idx = {}
    nan_idx = {}
    cnt_thresh = 0.75  # Only count a QoI as an outlier if more than 75% of its values are outliers; always true for scalars

    for var, arr in outputs.items():
        if COORDS_STR_ID in str(var):
            continue

        try:
            arr = _object_to_numeric(arr)
        except Exception:
            # Might have object arrays of different shapes, so instead estimate outliers for each sample individually
            if np.issubdtype(arr.dtype, np.object_):
                nan_idx.setdefault(var, np.full(num_samples, False))
                outlier_idx.setdefault(var, np.full(num_samples, False))
                for i, field_qty in enumerate(arr):
                    nan_idx[var][i] |= np.any(np.isnan(field_qty))

                    p25 = np.percentile(field_qty, 25)
                    p75 = np.percentile(field_qty, 75)
                    iqr = p75 - p25
                    outlier_idx[var][i] |= np.sum((field_qty < p25 - iqr_factor * iqr) |
                                                  (field_qty > p75 + iqr_factor * iqr)) > int(cnt_thresh * field_qty.size)
        else:
            # Directly use scalar numeric arrays (or obj arrays that were successfully converted)
            nan_idx.setdefault(var, np.full(num_samples, False))
            outlier_idx.setdefault(var, np.full(num_samples, False))

            nan_idx[var] |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))

            p25 = np.percentile(arr, 25, axis=0)
            p75 = np.percentile(arr, 75, axis=0)
            iqr = p75 - p25
            outlier_idx[var] |= np.sum((arr < p25 - iqr_factor * iqr) | (arr > p75 + iqr_factor * iqr),
                                       axis=tuple(range(1, arr.ndim))) > int(cnt_thresh * np.prod(arr.shape[1:]))

    return nan_idx, outlier_idx


def _get_discard_idx(nan_idx: dict,
                     outlier_idx: dict,
                     discard_outliers: bool = False,
                     warn_func: callable = None,
                     warn_description: str = ''):
    """Get the indices of samples to discard based on nans and outliers.

    :param nan_idx: a `dict` with boolean arrays of shape `(num_samples,)` indicating nan samples for each output
    :param outlier_idx: a `dict` with boolean arrays of shape `(num_samples,)` indicating outliers for each output
    :param discard_outliers: whether to discard outliers from the data. Defaults to False.
    :param warn_func: a function to use for warnings. No warnings by default.
    :param warn_description: a description to use in the warning message. Defaults to an empty string.
    :returns: a boolean array of shape `(num_samples,)` indicating samples to discard
    """
    num_samples = next(iter(nan_idx.values())).shape[0]
    discard_idx = np.full(num_samples, False)

    all_nan = np.full(num_samples, False)
    for var, idx in nan_idx.items():
        all_nan |= idx

    discard_idx |= all_nan  # always discard nan samples

    all_outliers = np.full(num_samples, False)
    for var, idx in outlier_idx.items():
        all_outliers |= idx

    if warn_func is not None:
        if np.any(all_nan):
            warn_func(f'Discarded {np.sum(all_nan)}/{num_samples} {warn_description} samples with nans.')
        if np.any(all_outliers):
            warn_func(f'Detected {np.sum(all_outliers)}/{num_samples} {warn_description} outliers.')

    if discard_outliers:
        if warn_func is not None:
            system.logger.warning(f'Discarding outliers...')
        discard_idx |= all_outliers

    return discard_idx


def generate_data(system: System, description: str,
                  num_samples: int = 500,
                  executor: Executor = None,
                  verbose: bool = False,
                  iqr_factor: float = 1.5):
    """Randomly sample the input space and compute the models for the `system` object. Will save the data to a `.pkl`
     file in the `system.root_dir/description` directory.

    :param system: the `amisc.System` object with the model and input/output variable information. The `system.root_dir`
                   must be set to the directory where the data will be saved.
    :param description: the description of the data being generated (e.g. 'compression', 'test_set', etc.), will be
                        used as an output directory and file name for all generated data.
    :param num_samples: the number of samples to use for generating the data. Defaults to 500.
    :param executor: the parallel executor for evaluating the models (i.e. a `concurrent.futures.Executor` instance)
    :param verbose: whether to print verbose output. Defaults to False.
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :returns: the `pickle` dump of the generated data (i.e. a `dict` with the data and indices of outliers/nans)
    """
    system.logger.info(f'Generating {description} data for {system.name} -- {num_samples} samples...')
    os.mkdir(Path(system.root_dir) / description)
    samples = system.sample_inputs(num_samples, normalize=True, use_pdf=['calibration', 'nuisance'])
    outputs = system.predict(samples, use_model='best', model_dir=Path(system.root_dir) / description,
                             executor=executor, verbose=verbose)

    samples, coords = to_model_dataset(samples, system.inputs())
    samples.update(coords)

    norm_outputs = {}
    for var in system.outputs():
        if var in outputs:
            norm_outputs[var.name] = var.normalize(_object_to_numeric(outputs[var.name]))

    # Get indices of bad samples and outliers (by interquartile range)
    nan_idx, outlier_idx = _filter_outputs(norm_outputs, iqr_factor=iqr_factor)

    with open(Path(system.root_dir) / description / f'{description}.pkl', 'wb') as fd:
        dump = {description: (samples, outputs), 'nan_idx': nan_idx, 'outlier_idx': outlier_idx,
                'iqr_factor': iqr_factor}
        pickle.dump(dump, fd)

    return dump


def process_compression(system: System, data: dict, discard_outliers: bool = False):
    """Compute compression maps for field quantities (only SVD supported). The compression parameters, such as
    rank, reconstruction tolerance, etc. should be specified in the config file that was used to load the `system`.

    Will save the compression maps to the `system.root_dir/compression` directory.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param data: the compression data from `generate_data` (i.e. a `dict` with the data and indices of outliers/nans)
    :param discard_outliers: whether to discard outliers from the compression data. Defaults to False.
    """
    outputs = data['compression'][1]
    nan_idx = data['nan_idx']
    outlier_idx = data['outlier_idx']

    discard_idx = _get_discard_idx(nan_idx, outlier_idx, discard_outliers=discard_outliers,
                                   warn_func=system.logger.warning, warn_description='compression')

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
                    var.compression.compute_map(data_matrix)
                case other:
                    raise ValueError(f"Compression method '{other}' not supported.")

    system.save_to_file(f'{system.name}_compression.yml', Path(system.root_dir) / 'compression')


def plot_outliers(outputs: dict, outlier_idx: dict, iqr_factor: float = 1.5, subplot_size: float = 3,
                  fields_1d: list[str] = None, fields_log: dict = None):
    """Plot histograms of outliers detected in the output data at the given indices.

    :param outputs: dictionary of output arrays, where each array has shape `(num_samples, ...)`.
    :param outlier_idx: dictionary with boolean arrays of shape `(num_samples,)` indicating outliers for each output
    :param iqr_factor: the factor to multiply the IQR by to determine outliers. Defaults to 1.5.
    :param subplot_size: the size of each subplot in inches. Defaults to 2.5.
    :param fields_1d: a list of field quantities to plot as 1D line plots. Defaults to `['u_ion', 'j_ion']`. Otherwise,
           all field quantities will be plotted as histograms of the mean value.
    :param fields_log: a dictionary of field qtys to set the y-axis to log scale. Defaults to {'j_ion': True}.
    :returns: the `fig, ax` for the plot
    """
    fields_1d = fields_1d or ['u_ion', 'j_ion']
    fields_log = fields_log or {'j_ion': True}
    num_plots = len(outlier_idx.keys())
    num_col = int(np.floor(np.sqrt(num_plots)))
    num_row = int(np.ceil(num_plots / num_col))

    fig, axs = plt.subplots(num_row, num_col, figsize=(num_col * subplot_size, num_row * subplot_size), layout='tight',
                            squeeze=False)
    for i, (var, idx) in enumerate(outlier_idx.items()):
        row = i // num_col
        col = i % num_col
        ax = axs[row, col]
        all_data = _object_to_numeric(outputs[var])
        outliers = all_data[idx, ...]

        p2 = np.nanpercentile(all_data, 2.5, axis=0)
        p25 = np.nanpercentile(all_data, 25, axis=0)
        p50 = np.nanpercentile(all_data, 50, axis=0)
        p75 = np.nanpercentile(all_data, 75, axis=0)
        p98 = np.nanpercentile(all_data, 97.5, axis=0)
        iqr = p75 - p25
        lb_iqr = p25 - iqr_factor * iqr
        ub_iqr = p75 + iqr_factor * iqr

        # Line plots for 1d field quantities
        if var in fields_1d:
            coords = np.linspace(0, 1, len(p50))
            ax.plot(coords, p50, '-k', label='Median')
            ax.fill_between(coords, lb_iqr, ub_iqr, alpha=0.5, edgecolor=(0.5, 0.5, 0.5), facecolor='gray',
                            label='IQR bounds')
            ax.plot(coords, p2, '-b', label='95% bounds')
            ax.plot(coords, p98, '-b')
            ax.plot(np.nan, np.nan, '--', label='Outliers')
            for arr in outliers:
                ax.plot(coords, arr, '--', alpha=0.3)
            ax.set_xlabel('Normalized field location')
            ax.set_ylabel(var)
            ax.grid()
            ax.legend()
            if var in fields_log:
                ax.set_yscale('log')

        # Histograms for everything else
        else:
            axes = tuple(range(1, outliers.ndim))
            ax.hist(all_data, bins=30, facecolor='gray', edgecolor='k', alpha=0.5)
            ax.hist(np.nanmean(outliers, axis=axes), facecolor='r', edgecolor='k', alpha=0.3, label='Outliers')
            ax.axvline(x=np.nanmean(p50, axis=axes), color='k', linestyle='-', linewidth=1.5, label='Median')
            ax.axvline(x=np.nanmean(p2, axis=axes), color='b', linestyle='-', linewidth=1.5, label='95% bounds')
            ax.axvline(x=np.nanmean(p98, axis=axes), color='b', linestyle='-', linewidth=1.5)
            ax.axvline(x=np.nanmean(lb_iqr, axis=axes), color='gray', linestyle='--', linewidth=1.5, label='IQR bounds')
            ax.axvline(x=np.nanmean(ub_iqr, axis=axes), color='gray', linestyle='--', linewidth=1.5)
            ax.set_xlabel(var)
            if row == 0 and col == num_col - 1:
                ax.legend()

    return fig, axs


def plot_compression(system: System, data: dict):
    """Generate plots of the compression data for field quantities.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param data: the compression data from `generate_data` (i.e. a `dict` with the data and indices of outliers/nans)
    """
    outputs = data['compression'][1]
    nan_idx = data['nan_idx']
    outlier_idx = data['outlier_idx']
    iqr_factor = data['iqr_factor']

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
                A = _object_to_numeric(outputs[var.name])
                lb = np.nanpercentile(A, 2.5, axis=0)
                mid = np.nanpercentile(A, 50, axis=0)
                ub = np.nanpercentile(A, 97.5, axis=0)

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

    fig, ax = plot_outliers(outputs, outlier_idx, iqr_factor=iqr_factor)
    fig.savefig(Path(system.root_dir) / 'compression' / 'outliers.png', dpi=300, format='png')


def plot_test_set(system: System, data: dict):
    """Plot results for the test set. Will save the plots to the `system.root_dir/test_set` directory.

    :param system: the `amisc.System` surrogate object with the model and input/output variable information.
    :param data: the test set data from `generate_data` (i.e. a `dict` with the data and indices of outliers/nans)
    """
    outputs = data['test_set'][1]
    nan_idx = data['nan_idx']
    outlier_idx = data['outlier_idx']
    iqr_factor = data['iqr_factor']

    discard_idx = _get_discard_idx(nan_idx, outlier_idx, discard_outliers=args.discard_outliers,
                                   warn_func=system.logger.warning, warn_description='test set')

    fig, ax = plot_outliers(outputs, outlier_idx, iqr_factor=iqr_factor)
    fig.savefig(Path(system.root_dir) / 'test_set' / 'outliers.png', dpi=300, format='png')


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

    with pool_executor(max_workers=args.gen_cpus) as executor:
        compression_data = generate_data(system, 'compression', num_samples=args.compression_samples,
                                         executor=executor, verbose=True, iqr_factor=args.iqr_factor)
        test_set_data = generate_data(system, 'test_set', num_samples=args.test_samples,
                                      executor=executor, verbose=True, iqr_factor=args.iqr_factor)

    process_compression(system, compression_data, discard_outliers=args.discard_outliers)

    plot_compression(system, compression_data)
    plot_test_set(system, test_set_data)

    system.logger.info('Data generation complete.')
