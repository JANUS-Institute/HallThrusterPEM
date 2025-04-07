import argparse
import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pem_mcmc.analysis import COLORS, RCPARAMS, save_figure

plt.rcParams.update(RCPARAMS)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--prior", type=str, required=True)
parser.add_argument("--posterior", type=str, required=True)
parser.add_argument("--out", type=str, default=".")
parser.add_argument("--loglevel", type=int, default=logger.level)
parser.add_argument(
    "--marker-scale", type=float, default=1.0, help="How much the marker radii should be scaled from the default value."
)
parser.add_argument("--alpha", type=float, default=0.5, help="How opaquely the points should be drawn.")

QUANTITY_NAMES = {
    "thrust_N": "thrust [mN]",
    "discharge_current_A": "discharge current [A]",
    "cathode_coupling_voltage_V": "cathode coupling voltage [V]",
}


@dataclass
class Dataset:
    x_mean: np.ndarray
    x_err: np.ndarray
    y_median: np.ndarray
    y_err_lo: np.ndarray
    y_err_hi: np.ndarray


def load_dataset(folder: PathLike, quantity: str) -> Dataset | None:
    file = Path(folder) / "mcmc_analysis" / (quantity + ".csv")

    if not file.exists():
        logger.debug(f"File {file} not found. Skipping.")
        return None

    data = np.genfromtxt(file, delimiter=",")
    num_cols = data.shape[1]
    if num_cols != 5:
        raise ValueError(f"Expected 5 columns in data file '{file}' but found {num_cols}.")

    return Dataset(
        x_mean=data[:, 0],
        x_err=data[:, 1],
        y_median=data[:, 2],
        y_err_lo=data[:, 3],
        y_err_hi=data[:, 4],
    )


BASE_ZORDER = 5


def _plot_multiple_datasets(
    ax: Axes,
    data: list[Dataset],
    colors: list[str],
    markers: list[str],
    markersizes: list[float],
    alpha: float,
) -> tuple[list, tuple[float, float], tuple[float, float]]:
    handles = []

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i, (_data, _color, _marker, _markersize) in enumerate(zip(data, colors, markers, markersizes)):
        y_err = (_data.y_err_lo, _data.y_err_hi)
        h_err = ax.errorbar(
            _data.x_mean,
            _data.y_median,
            xerr=_data.x_err,
            yerr=y_err,
            color=COLORS[_color],
            fmt='none',
            alpha=alpha,
            zorder=BASE_ZORDER + 2 * i,
        )

        h_marker = ax.scatter(
            _data.x_mean,
            _data.y_median,
            s=(_markersize) ** 2,
            color=COLORS[_color],
            alpha=1.0,
            linewidth=0,
            zorder=BASE_ZORDER + 2 * i + 1,
            marker=_marker,
        )
        handles.append((h_err, h_marker))
        xmin = min(xmin, np.min(_data.x_mean - _data.x_err))
        xmax = max(xmax, np.max(_data.x_mean + _data.x_err))
        ymin = min(ymin, np.min(_data.y_median - _data.y_err_lo))
        ymax = max(ymax, np.max(_data.y_median + _data.y_err_hi))

    return handles, (xmin, xmax), (ymin, ymax)


def pad_lims(lims: tuple[float, float], pad: float) -> tuple[float, float]:
    lo, hi = lims
    diff = pad * (hi - lo)
    return lo - diff, hi + diff


def plot_prediction(
    prior: Dataset, posterior: Dataset, quantity: str, output_path: Path, marker_scale: float, alpha: float
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

    data = [prior, posterior]
    labels = ['Prior', 'Posterior']
    colors = ['lightorange', 'blue']
    markers = ['o', '^']
    markersizes = [4.0, 5.0]
    markersizes = [size * marker_scale for size in markersizes]

    handles, xlims, _ = _plot_multiple_datasets(ax, data, colors, markers, markersizes, alpha)
    xlims = pad_lims(xlims, 0.1)
    vals = list(xlims)

    ax.plot(vals, vals, color='black', zorder=BASE_ZORDER + 2 * len(data) + 1, linestyle='--', linewidth=2)

    qty_name = QUANTITY_NAMES[quantity]
    ax.set_xlabel("Experimental " + qty_name)
    ax.set_ylabel("Predicted " + qty_name)
    ax.set_xlim(xlims)

    ax.legend(handles, labels, loc='upper left', markerscale=2.0)

    save_figure(fig, output_path, quantity)


def try_plot_prediction(
    prior: PathLike, posterior: PathLike, quantity: str, output_path: Path, marker_scale: float, alpha: float
) -> None:
    data_prior = load_dataset(prior, quantity)
    data_posterior = load_dataset(posterior, quantity)
    if data_prior is None or data_posterior is None:
        return

    logger.debug(f"Plotting {quantity}")
    plot_prediction(data_prior, data_posterior, quantity, output_path, marker_scale, alpha)


def main(args):
    quantities = ["thrust_N", "discharge_current_A", "cathode_coupling_voltage_V"]
    for quantity in quantities:
        try_plot_prediction(args.prior, args.posterior, quantity, Path(args.out), args.marker_scale, args.alpha)


if __name__ == "__main__":
    args = parser.parse_args()

    logger.setLevel(args.loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(args.loglevel)
    ch.setFormatter(logging.Formatter("%(levelname)s:$(name):\t%(message)s"))
    logger.addHandler(ch)

    main(args)
