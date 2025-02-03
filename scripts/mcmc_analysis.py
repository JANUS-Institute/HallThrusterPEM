import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional, TypeAlias

import ash
import matplotlib.pyplot as plt
import numpy as np
from amisc import System
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist, euclidean

import hallmd.data
from hallmd.data import OperatingCondition, ThrusterData, spt100

Dataset: TypeAlias = dict[OperatingCondition, ThrusterData]

plt.rcParams.update({"font.size": '14'})


def mean_and_median(X, eps=1e-5):
    # https://stackoverflow.com/a/30305181/22854790
    mean_y = np.mean(X, 0)
    y = mean_y

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return mean_y, y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return mean_y, y1

        y = y1


def start_timer(msg: str) -> float:
    print(msg, "... ", end="")
    return time.time()


def stop_timer(start_time: float):
    elapsed = time.time() - start_time
    print(f"took {elapsed:.2f} s.")


def analyze_mcmc(path, config, datasets, corner=False, bands=False):
    mcmc_path = Path(path) / "mcmc"
    logfile = mcmc_path / "mcmc.csv"
    plot_path = Path(path) / "mcmc_analysis"
    os.makedirs(plot_path, exist_ok=True)
    print("Generating plots in", plot_path)

    analysis_start = time.time()

    system = System.load_from_file(Path(path) / config)

    dlm = ","

    with open(logfile, "r") as file:
        header = file.readline().rstrip()

        fields = header.split(dlm)
        var_start = 1
        var_end = len(fields) - 2

        id_ind = 0
        variables = fields[var_start:var_end]
        log_post_ind = len(variables) + 1
        accept_ind = len(variables) + 2

        assert fields[id_ind] == "id"
        assert fields[log_post_ind] == "log_posterior"
        assert fields[accept_ind] == "accepted"

        samples = []
        logposts = []
        accepted = []
        ids = []
        total_samples = 0

        for line in file:
            total_samples += 1
            fields = line.rstrip().split(dlm)
            accept_str = fields[accept_ind].casefold()

            if accept_str == "true":
                accept = True
            elif accept_str == "false":
                accept = False
            else:
                raise ValueError(f"Invalid accept value {fields[accept_ind]} at row {total_samples} in file {logfile}.")

            if accept:
                ids.append(fields[id_ind])
            else:
                ids.append(ids[-1])

            logposts.append(float(fields[log_post_ind]))
            samples.append(np.array([float(x) for x in fields[var_start:var_end]]))
            accepted.append(accept)

    burn_in_frac = 0.5
    num_burn = math.floor(burn_in_frac * len(samples))

    samples_raw = np.array(samples)
    samples = np.array(samples)[num_burn:]
    logposts = np.array(logposts)
    map_ind = np.argmax(logposts)
    accepted = np.array(accepted)[num_burn:]

    num_accept = len(samples)
    tex_names = [system.inputs()[name].tex for name in variables]

    if num_accept > 10:
        start = start_timer("Computing map, mean, median, and covariance")
        map = samples_raw[map_ind, :]
        mean, median = mean_and_median(samples)
        empirical_covariance = np.cov(samples.T)
        header = ",".join(variables)
        np.savetxt(plot_path / "map.csv", map, header=header)
        np.savetxt(plot_path / "mean.csv", mean, header=header)
        np.savetxt(plot_path / "median.csv", median, header=header)
        np.savetxt(plot_path / "cov.csv", empirical_covariance, header=header)
        stop_timer(start)

        start = start_timer("Plotting traces")
        start = time.time()
        plot_traces(tex_names, samples_raw, plot_path)
        stop_timer(start)

        if corner:
            start = start_timer("Plotting corner plot")
            plot_corner(samples, tex_names, plot_path, map=map, mean=mean, median=median)
            stop_timer(start)

    start = start_timer("Loading data")
    data = hallmd.data.load(spt100.datasets_from_names(datasets))
    map = load_sim_results([ids[map_ind]], mcmc_path)[0]['output']
    stop_timer(start)

    # Plot bands
    if bands:
        start = start_timer("Loading results")
        results_all = load_sim_results(ids[num_burn:], mcmc_path)
        outputs = [res['output'] for res in results_all]
        stop_timer(start)

        start = start_timer("Plotting thrust")
        plot_global_quantity(data, outputs, plot_path, "thrust_N", "Thrust [mN]", map=map, scale=1000, lims=(0, 110))
        stop_timer(start)

        start = start_timer("Plotting current")
        plot_global_quantity(
            data, outputs, plot_path, "discharge_current_A", "Discharge current [A]", map=map, lims=(0, 7.5)
        )
        stop_timer(start)

        start = start_timer("Plotting cathode coupling voltage")
        plot_global_quantity(
            data,
            outputs,
            plot_path,
            "cathode_coupling_voltage_V",
            "Cathode coupling voltage [V]",
            map=map,
            lims=(0, 40),
        )
        stop_timer(start)

        start = start_timer("Plotting ion velocity")
        plot_field_quantity(
            data,
            outputs,
            plot_path,
            map=map,
            xquantity="ion_velocity_coords_m",
            yquantity="ion_velocity_m_s",
            xlabel="Axial coordinate [channel lengths]",
            ylabel="Ion velocity [km/s]",
            xscalefactor=1 / 0.025,
            yscalefactor=1 / 1000,
        )
        stop_timer(start)

        start = start_timer("Plotting ion current density")
        plot_field_quantity(
            data,
            outputs,
            plot_path,
            map=map,
            xquantity="ion_current_density_coords_rad",
            yquantity="ion_current_density_A_m2",
            xlabel="Angle [degrees]",
            ylabel="Ion current density [A/m$^2$]",
            xscalefactor=180 / np.pi,
            yscalefactor=1,
            yscale='log',
        )
        stop_timer(start)

    plt.close('all')

    print(f"Analysis finished in {time.time() - analysis_start:.2f} s.")


def _extract_quantity(data: Dataset, quantity: str, sorted=False):
    mask = np.array([getattr(x, quantity) is not None for x in data.values()])
    pressure = np.array([opcond.background_pressure_torr for opcond in data.keys()])[mask] * 1e6
    qty = np.array([getattr(x, quantity) for i, x in enumerate(data.values()) if mask[i]])

    if sorted:
        perm = np.argsort(pressure)
        return pressure[perm], qty[perm]
    else:
        return pressure, qty


def plot_field_quantity(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    xquantity: str,
    yquantity: str,
    xlabel: str,
    ylabel: str,
    xscalefactor: float,
    yscalefactor: float,
    map: Dataset | None = None,
    yscale: str = 'linear',
):
    out_path = plot_path / yquantity
    os.makedirs(out_path, exist_ok=True)

    mask = np.array([getattr(x, yquantity) is not None for x in data.values()])
    opconds = [opcond for (i, opcond) in enumerate(data.keys()) if mask[i]]

    if len(opconds) == 0:
        return

    colors = plt.get_cmap('turbo')

    # Extract simulation coords and data
    medians = {}

    # Individual plots for each pressure
    for i, opcond in enumerate(opconds):
        _data = data[opcond]
        pressure_uTorr = opcond.background_pressure_torr * 1e6

        fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.autoscale(enable=True, tight=True)
        ax.set_title(ylabel + f" ($p_B = {pressure_uTorr:.0f}$ $\\mu$Torr)")

        x_data = getattr(_data, xquantity) * xscalefactor
        y_data = getattr(_data, yquantity)
        y_data_mean = y_data.mean * yscalefactor
        y_data_std = y_data.std * yscalefactor

        ax.errorbar(x_data, y_data_mean, yerr=2 * y_data_std, color="black", capsize=5, linestyle="")
        ax.scatter(x_data, y_data_mean, s=40, color='black', label="Data", zorder=10)

        x_sim = getattr(sim[0][opcond], xquantity) * xscalefactor
        y_sim = np.array([getattr(_sim[opcond], yquantity).mean * yscalefactor for _sim in sim])

        qt = _plot_quantiles(ax, x_sim, y_sim)
        medians[opcond] = qt[2]

        if map is not None:
            x_map = getattr(map[opcond], xquantity) * xscalefactor
            y_map = getattr(map[opcond], yquantity).mean * yscalefactor
            ax.plot(x_map, y_map, label="Best sample", color='red')

        handles, labels = ax.get_legend_handles_labels()
        ax_legend.legend(handles, labels, borderaxespad=0)
        ax_legend.axis("off")
        fig.tight_layout()

        plot_name = f"{yquantity}_p={pressure_uTorr}uTorr.png"
        fig.savefig(out_path / plot_name)
        plt.close(fig)

    # Median predictions and best sample vs data on one plot
    fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.autoscale(enable=True, tight=True)

    for i, opcond in enumerate(opconds):
        _data = data[opcond]
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        color = colors(1 - (i + 0.5) / len(opconds))
        x_data = getattr(_data, xquantity) * xscalefactor
        y_data = getattr(_data, yquantity)
        y_data_mean = y_data.mean * yscalefactor
        y_data_std = y_data.std * yscalefactor
        ax.errorbar(x_data, y_data_mean, yerr=2 * y_data_std, color=color, capsize=3, linestyle="")
        ax.plot(
            x_data, y_data_mean, '--o', color=color, label=f"Data ($p_B = {pressure_uTorr:.0f}$ $\\mu$Torr)", zorder=10
        )

        x_sim = getattr(sim[0][opcond], xquantity) * xscalefactor
        ax.plot(x_sim, medians[opcond], color=color, label="Median prediction")

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()
    plot_name = f"{yquantity}_allpressures.png"
    fig.savefig(out_path / plot_name)
    plt.close(fig)


def plot_global_quantity(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    quantity: str,
    full_name: str,
    map: Dataset | None = None,
    scale: float = 1,
    lims=None,
):
    xlabel = "Background pressure [$\\mu$Torr]"
    pressure_data, qty_data = _extract_quantity(data, quantity)

    if len(qty_data) == 0:
        return

    qty_data_mean = np.array([x.mean for x in qty_data]) * scale
    qty_data_std = np.array([x.std for x in qty_data]) * scale

    mask_sim = np.array([getattr(x, quantity) is not None for x in sim[0].values()])
    pressure_sim = np.array([opcond.background_pressure_torr for opcond in sim[0].keys()])[mask_sim] * 1e6
    qty_sim = np.array(
        [[getattr(x, quantity).mean * scale for i, x in enumerate(_sim.values()) if mask_sim[i]] for _sim in sim]
    )

    sortperm_sim = np.argsort(pressure_sim)
    pressure_sim = pressure_sim[sortperm_sim]
    qty_sim = qty_sim[:, sortperm_sim]

    fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
    if lims is not None:
        ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(full_name)
    ax.set_xlim(0, np.max(pressure_sim) * 1.05)

    ax.errorbar(pressure_data, qty_data_mean, yerr=2 * qty_data_std, color="black", capsize=5, linestyle="")
    ax.scatter(pressure_data, qty_data_mean, s=40, color='black', label="Data", zorder=10)

    _plot_quantiles(ax, pressure_sim, qty_sim)

    if map is not None:
        pressure_map, qty_map = _extract_quantity(map, quantity)
        qty_map_mean = np.array([x.mean * scale for x in qty_map])
        ax.scatter(pressure_map, qty_map_mean, s=64, marker='x', color='red', label="Best sample", zorder=9)

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()

    fig.savefig(plot_path / f"{quantity}_pressure_bands.png")
    plt.close(fig)


def load_sim_results(ids, mcmc_path: Path) -> list[dict]:
    data = []
    for id in ids:
        amisc_path = mcmc_path / id / "pemv1.pkl"
        with open(amisc_path, "rb") as f:
            data.append(pickle.load(f))

    return data


def _plot_quantiles(ax, x, y, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    qt = np.quantile(y, q=quantiles, axis=0)
    lc = (0.3, 0.3, 0.3)
    outer_args = {'linestyle': ':', 'color': lc, 'zorder': 2}
    inner_args = {'linestyle': '-.', 'color': lc, 'zorder': 2}
    ax.plot(x, qt[2], color=lc, linestyle='--', linewidth=2, label="Median prediction", zorder=2)
    ax.fill_between(x, qt[1, :], qt[-2, :], facecolor=(0.7, 0.7, 0.7), label='50% CI', zorder=1)
    ax.fill_between(x, qt[0, :], qt[-1, :], facecolor=(0.9, 0.9, 0.9), label='90% CI', zorder=0)
    ax.plot(x, qt[0], **outer_args)
    ax.plot(x, qt[-1], **outer_args)
    ax.plot(x, qt[1], **inner_args)
    ax.plot(x, qt[-2], **inner_args)

    return qt


def plot_result(data, result, id, plot_path):
    input = result['input']
    output = result['output']
    plot_u_ion(data, output, 0.025, id, plot_path)
    plot_j_ion(data, output, id, plot_path)
    # plot_global_quantities(data, output, id, plot_path)
    return


def plot_j_ion(data: Dataset, output: Dataset | None, plot_name: str, plot_path: Path):
    fig, (ax, ax_legend) = plt.subplots(1, 2, width_ratios=[2.5, 1], figsize=(10, 6))

    ax.set_xlim(0, 90)
    ax.set_yscale('log')
    ax.set_xlabel("Probe angle [deg]")
    ax.set_ylabel("$j_{ion}$ [A / m$^2$]")

    colors = plt.get_cmap('turbo')

    rad2deg = 180 / np.pi

    color_ind = 0
    for i, (opcond, dataset) in enumerate(data.items()):
        color_scale = color_ind / 8.0
        pressure_base = round(1e5 * opcond.background_pressure_torr, 1)

        theta_rad = dataset.ion_current_density_coords_rad
        j_ion = dataset.ion_current_density_A_m2

        if theta_rad is None or j_ion is None:
            continue

        theta_deg = theta_rad * rad2deg

        label = f"$P_B = {pressure_base}\\times 10^{{-5}}$ Torr"
        ax.errorbar(
            theta_deg, j_ion.mean, yerr=2 * j_ion.std, label=label, fmt="--o", markersize=4.0, color=colors(color_scale)
        )

        if output is not None and opcond in output:
            result = output[opcond]
            theta_rad_sim, j_sim = result.ion_current_density_coords_rad, result.ion_current_density_A_m2

            if theta_rad_sim is not None and j_sim is not None:
                ax.plot(theta_rad_sim * rad2deg, j_sim.mean, color=colors(color_scale))

        color_ind += 1

    if color_ind == 0:
        # nothing was plotted
        return

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path / f"j_ion_{plot_name}.png", dpi=300)
    plt.close(fig)

    return


def plot_u_ion(data: Dataset, output: Dataset | None, L_ch: float, plot_name: str, plot_path: Path):
    colors = ["red", "green", "blue"]
    fig, (ax, ax_legend) = plt.subplots(1, 2, width_ratios=[2.5, 1], figsize=(10, 6))
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("$z / L_{ch}$")
    ax.set_ylabel("$u_{ion, z}$ [km/s]")

    color_ind = 0
    for i, (opcond, dataset) in enumerate(data.items()):
        pressure_base = round(1e5 * opcond.background_pressure_torr, 1)

        z = dataset.ion_velocity_coords_m
        u_ion = dataset.ion_velocity_m_s

        if z is None or u_ion is None:
            continue

        label = f"$P_B = {pressure_base}\\times 10^{{-5}}$ Torr"
        ax.errorbar(
            z / L_ch,
            u_ion.mean / 1000,
            yerr=2 * u_ion.std / 1000,
            label=label,
            color=colors[color_ind],
            fmt="--o",
            markersize=4.0,
        )

        if output is not None:
            result = output[opcond]
            z_sim, u_sim = result.ion_velocity_coords_m, result.ion_velocity_m_s

            if z_sim is not None and u_sim is not None:
                ax.plot(z_sim / L_ch, u_sim.mean / 1000, color=colors[color_ind])

        color_ind += 1

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path / f"u_ion_{plot_name}.png", dpi=300)
    plt.close(fig)

    return


def plot_traces(names, samples, dir: Path = Path(".")):
    _, num_vars = samples.shape

    max_rows = 6
    if num_vars > max_rows:
        num_rows = max_rows
        num_cols = math.ceil(num_vars / max_rows)
    else:
        num_rows = num_vars
        num_cols = 1

    subfig_height = 2
    subfig_width = 8
    fig_kw = {"figsize": (subfig_width * num_cols, subfig_height * num_rows), "dpi": 200}

    fig, axes = plt.subplots(num_rows, num_cols, **fig_kw, sharex='col')

    perm = [i for i, _ in sorted(enumerate(names), key=lambda x: x[1])]

    N = samples.shape[0]

    for col in range(num_cols):
        for row in range(num_rows):
            ax = axes[row, col]
            i = row + num_rows * col
            if i >= len(perm):
                continue
            index = perm[row + num_rows * col]

            if row == num_rows - 1:
                ax.set_xlabel("Iteration")
            ax.set_ylabel(names[index])
            ax.set_xlim(0, N)
            ax.plot(np.arange(N), samples[:, index], color='black')

    plt.tight_layout()
    outpath = dir / "traces.png"
    fig.savefig(outpath)
    plt.close(fig)
    return


def plot_corner(
    samples: np.ndarray,
    var_names: list[str],
    dir: Path = Path("."),
    map: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    median: np.ndarray | None = None,
) -> tuple[Figure, Axes]:
    fontsize = 12
    num_vars = len(var_names)
    size = 3 * num_vars
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(size, size))

    lims = [_determine_limits(samples[:, i]) for i in range(num_vars)]

    map_color = "red"
    mean_color = "orange"
    median_color = "pink"

    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            if i > j:
                ax.set_axis_off()
                continue
            elif i == j:
                _ax_hist1d(ax, samples[:, i], lims[i])
                ax.set_yticks([])

                if mean is not None:
                    ax.axvline(mean[i], color=mean_color, linewidth=2)

                if median is not None:
                    ax.axvline(median[i], color=median_color, linewidth=2)

                if map is not None:
                    ax.axvline(map[i], color=map_color, linewidth=2)

                if j == num_vars - 1:
                    ax.set_xlabel(var_names[i], fontsize=fontsize)
                else:
                    ax.set_xticklabels([])

            elif i < j:
                x = samples[:, i]
                y = samples[:, j]
                nbins_x = _num_bins(x, lims[i])
                nbins_y = _num_bins(y, lims[j])
                ax.hist2d(samples[:, i], samples[:, j], bins=[nbins_x, nbins_y], range=[lims[i], lims[j]])

                if mean is not None:
                    ax.scatter([mean[i]], [mean[j]], color=mean_color)

                if median is not None:
                    ax.scatter([median[i]], [median[j]], color=median_color)

                if map is not None:
                    ax.scatter([map[i]], [map[j]], color=map_color)

                if i == 0:
                    ax.set_ylabel(var_names[j], fontsize=fontsize)
                else:
                    ax.set_yticklabels([])

                if j == num_vars - 1:
                    ax.set_xlabel(var_names[i], fontsize=fontsize)
                else:
                    ax.set_xticklabels([])

            ax.tick_params(axis="x", rotation=-45)

    plt.tight_layout()
    fig.savefig(dir / "corner.png", dpi=100)
    plt.close(fig)
    return fig, axes


def _num_bins(samples: np.ndarray, lims: tuple[float, float]) -> int:
    """Calculate optimal histogram bin count using Friedman-Draconis rule"""
    q3 = np.percentile(samples, 75)
    q1 = np.percentile(samples, 25)
    iqr = q3 - q1

    bin_width = 2 * iqr / np.cbrt(samples.size)
    num_bins = np.ceil((lims[1] - lims[0]) / bin_width).astype(int)
    return num_bins


def _ax_hist1d(ax: Axes, samples: np.ndarray, xlims: tuple[float, float]) -> None:
    ax.set_xlim(xlims)
    nbins = _num_bins(samples, xlims)
    nshifts = 5
    bins, heights = ash.ash1d(samples, nbins, nshifts, range=xlims)
    ax.hist(samples, nbins, color="lightgrey", density=True)
    ax.plot(bins, heights, zorder=2, color="black", linewidth=2)


def _ax_hist2d(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    logpdfs: Optional[np.ndarray] = None,
) -> None:
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    nbins_x = _num_bins(x, xlims)
    nbins_y = _num_bins(y, ylims)
    nbins = min(nbins_x, nbins_y)
    nshifts = 5
    grid, heights = ash.ash2d(x, y, nbins, nshifts, xrange=xlims, yrange=ylims)

    scatter_kwargs = {
        "s": 6**2,
        "alpha": 1 / np.log10(x.size),
        "zorder": 1,
    }

    if logpdfs is None:
        ax.scatter(x, y, color="black", **scatter_kwargs)
    else:
        ax.scatter(x, y, c=logpdfs, **scatter_kwargs)

    ax.contour(grid[0], grid[1], heights, zorder=0)


def _determine_limits(x: np.ndarray) -> tuple[float, float]:
    min = np.min(x)
    max = np.max(x)
    return min, max


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MCMC analysis")

    parser.add_argument("path", type=str, help="the path to the directory containing the mcmc data")

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["diamant2014", "macdonald2019"],
        help="A list of datasets to use, pick from [diamant2014, macdonald2019, sankovic1993]",
    )

    parser.add_argument("--plot-corner", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--plot-bands", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config_file = ""
    for file in os.listdir(args.path):
        if file.endswith(".yml") or file.endswith(".yaml"):
            config_file = file

    if config_file == "":
        raise FileNotFoundError(f"No YAML file found in {args.path}.")

    analyze_mcmc(Path(args.path), Path(config_file), args.datasets, corner=args.plot_corner, bands=args.plot_bands)
