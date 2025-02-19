import colorsys
import math
import os
import pickle
import time
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from amisc import System
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist, euclidean

import hallmd.data
import hallmd.utils
from hallmd.data import OperatingCondition, ThrusterData

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


def analyze_mcmc(path, config, datasets, corner=False, bands=False, proposal_cov=None):
    mcmc_path = Path(path) / "mcmc"
    logfile = mcmc_path / "mcmc.csv"
    plot_path = Path(path) / "mcmc_analysis"
    os.makedirs(plot_path, exist_ok=True)
    print("Generating plots in", plot_path)

    analysis_start = time.time()

    system = System.load_from_file(Path(path) / config)
    device_name = system['Thruster'].model_kwargs['thruster']
    device = hallmd.utils.load_device(device_name)

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
        header = ",".join(variables)
        np.savetxt(plot_path / "map.csv", np.matrix(map), header=header, delimiter=',')
        np.savetxt(plot_path / "mean.csv", np.matrix(mean), header=header, delimiter=',')
        np.savetxt(plot_path / "median.csv", np.matrix(median), header=header, delimiter=',')

        # Save covariance passed to this function if present, otherwise compute empirical covariance and save it
        if proposal_cov is not None:
            np.savetxt(plot_path / "cov.csv", proposal_cov, header=header, delimiter=',')
        else:
            empirical_covariance = np.cov(samples.T)
            np.savetxt(plot_path / "cov.csv", empirical_covariance, header=header, delimiter=',')

        # Save last sample. Combined with proposal covariance, this lets us restart the MCMC process
        last = samples_raw[-1, :]
        np.savetxt(plot_path / "lastsample.csv", np.matrix(last), header=header, delimiter=",")

        stop_timer(start)

        start = start_timer("Plotting traces")
        start = time.time()
        plot_traces(tex_names, samples_raw, plot_path)
        stop_timer(start)

        if corner:
            try:
                start = start_timer("Plotting corner plot")
                plot_corner(samples, tex_names, plot_path, map=map, mean=mean, median=median)
                stop_timer(start)
            except ValueError:
                # the corner plot sometimes fails early on if there aren't enough distinct samples
                pass

    start = start_timer("Loading data")
    data = hallmd.data.load(hallmd.data.thrusters[device_name].datasets_from_names(datasets))
    channel_length = device['geometry']['channel_length']
    map = load_sim_results([ids[map_ind]], mcmc_path)[0]['output']
    stop_timer(start)

    if device_name == "SPT-100":
        thrust_lims = (0, 110)
        current_lims = (0, 7.5)
        vcc_lims = (0, 40)
    else:
        thrust_lims = (0, 400)
        current_lims = (0, 30)
        vcc_lims = (0, 40)

    # Plot bands
    if bands:
        start = start_timer("Loading results")
        results_all = load_sim_results(ids[num_burn:], mcmc_path)
        outputs = [res['output'] for res in results_all]
        stop_timer(start)

        start = start_timer("Plotting thrust")
        plot_global_quantity(data, outputs, plot_path, "thrust_N", "Thrust [mN]", map=map, scale=1000, lims=thrust_lims)
        stop_timer(start)

        start = start_timer("Plotting current")
        plot_global_quantity(
            data, outputs, plot_path, "discharge_current_A", "Discharge current [A]", map=map, lims=current_lims
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
            lims=vcc_lims,
        )
        stop_timer(start)

        start = start_timer("Plotting ion velocity")
        plot_ion_vel(
            data,
            outputs,
            plot_path,
            map=map,
            xlabel="Axial coordinate [channel lengths]",
            ylabel="Ion velocity [km/s]",
            xscalefactor=1 / channel_length,
            yscalefactor=1 / 1000,
        )
        stop_timer(start)

        start = start_timer("Plotting ion current density")
        plot_ion_cur(
            data,
            outputs,
            plot_path,
            map=map,
            xlabel="Angle [degrees]",
            ylabel="Ion current density [A/m$^2$]",
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


def _finalize_plot(fig, ax, ax_legend, out_path, plot_name):
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()
    fig.savefig(out_path / plot_name)
    plt.close(fig)


QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp(x, x0, x1):
    return max(min(x, x1), x0)


def darken(color, factor):
    hsv = colorsys.rgb_to_hsv(float(color[0]), float(color[1]), float(color[2]))
    return colorsys.hsv_to_rgb(clamp(hsv[0], 0, 1), clamp(hsv[1] * 0.75, 0, 1), clamp(hsv[2] * factor, 0, 1))


def plot_ion_cur(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    xlabel: str,
    ylabel: str,
    map: Dataset | None = None,
):
    qty_name = "ion_current_density"
    out_path = plot_path / qty_name
    os.makedirs(out_path, exist_ok=True)

    # get the sweep radii
    pressures, _ = _extract_quantity(data, "ion_current_sweeps", sorted=True)
    _, sim_data_0 = _extract_quantity(sim[0], "ion_current_sweeps", sorted=True)
    sweep_radii = [x.radius_m for x in sim_data_0[0]]
    angles_sim = sim_data_0[0][0].angles_rad * 180 / np.pi
    opconds = list(data.keys())

    colors = plt.get_cmap('turbo')
    xlims = (0, 90)
    yscale = 'log'

    data_kwargs = {'fmt': ':o', 'markersize': 4, 'alpha': 1}

    incr = 2

    # for each pressure, plot predictions at all radii
    jion_quantiles = {}
    for i, opcond in enumerate(opconds):
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        _data = data[opcond].ion_current_sweeps
        if _data is None or len(data) == 0:
            continue

        jion_quantiles[opcond] = []

        fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlims)
        ax.set_title(ylabel + f" ($p_B = {pressure_uTorr:.2f}$ $\\mu$Torr)")

        # plot data for this operating condition at each radius
        for j, sweep in enumerate(_data):
            color = colors((j + 0.5) / len(sweep_radii))
            theta_deg = sweep.angles_rad * 180 / np.pi
            jion = sweep.current_density_A_m2
            ax.errorbar(
                theta_deg[::incr],
                jion.mean[::incr],
                yerr=2 * jion.std[::incr],
                color=darken(color, 0.8),
                label=f"$r =$ {sweep.radius_m:.2f} m",
                **data_kwargs,
            )

            jion_sim = [_sim[opcond].ion_current_sweeps[j].current_density_A_m2.mean for _sim in sim]
            qt = np.quantile(jion_sim, q=QUANTILES, axis=0)
            # Plot median prediction
            ax.plot(angles_sim, qt[2], color=color, label="Median prediction")

            # _plot_quantiles(ax, angles_sim, qt, color=color, label=False)
            jion_quantiles[opcond].append(qt)

        plot_name = f"{qty_name}_p={pressure_uTorr:05.2f}uTorr.png"
        _finalize_plot(fig, ax, ax_legend, out_path, plot_name)

    # for each radius, plot predictions at each pressure
    for i, sweep_radius in enumerate(sweep_radii):
        fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlims)
        ax.set_title(ylabel + f"($r = {sweep_radius:.2f}$ m)")
        plot_name = f"{qty_name}_r={sweep_radius:.2f}m.png"

        # plot data for this operating condition at each pressure
        for j, opcond in enumerate(opconds):
            pressure_uTorr = opcond.background_pressure_torr * 1e6
            _data = data[opcond].ion_current_sweeps
            if _data is None or len(data) == 0:
                continue

            # make dedicated plot for each opcond
            fig_solo, (ax_solo, ax_legend_solo) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
            ax_solo.set_yscale(yscale)
            ax_solo.set_xlabel(xlabel)
            ax_solo.set_ylabel(ylabel)
            ax_solo.set_xlim(xlims)
            ax_solo.set_title(ylabel + f"($r = {sweep_radius:.2f}$ m)")
            pressure_str = f"$p_B = {pressure_uTorr:.2f}$ $\\mu$Torr"
            subdir = out_path / f"p={pressure_uTorr:05.2f}uTorr"
            os.makedirs(subdir, exist_ok=True)

            for k, axis in enumerate([ax, ax_solo]):
                sweep = _data[i]
                color = colors((j + 0.5) / len(pressures))
                theta_deg = sweep.angles_rad * 180 / np.pi
                jion = sweep.current_density_A_m2
                axis.errorbar(
                    theta_deg[::incr],
                    jion.mean[::incr],
                    yerr=2 * jion.std[::incr],
                    color=darken(color, 0.8) if k == 0 else 'black',
                    label=pressure_str,
                    **data_kwargs,
                )
                qt = jion_quantiles[opcond][i]
                if k == 0:
                    axis.plot(angles_sim, qt[2], color=color, label="Median prediction")
                else:
                    _plot_quantiles(axis, angles_sim, qt, color=(0.3, 0.3, 0.3), label=True)

            _finalize_plot(fig_solo, ax_solo, ax_legend_solo, subdir, plot_name)

        _finalize_plot(fig, ax, ax_legend, out_path, plot_name)

    return


def plot_ion_vel(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    xlabel: str,
    ylabel: str,
    xscalefactor: float,
    yscalefactor: float,
    map: Dataset | None = None,
):
    qty_name = "ion_velocity"
    out_path = plot_path / qty_name
    os.makedirs(out_path, exist_ok=True)

    mask = np.array([getattr(x, qty_name) is not None for x in data.values()])
    opconds = [opcond for (i, opcond) in enumerate(data.keys()) if mask[i]]

    if len(opconds) == 0:
        return

    colors = plt.get_cmap('turbo')

    # Extract simulation coords and data
    medians = {}
    data_kwargs = {'markersize': 4, 'capsize': 2}

    # Individual plots for each pressure
    for i, opcond in enumerate(opconds):
        _data = data[opcond].ion_velocity
        pressure_uTorr = opcond.background_pressure_torr * 1e6

        fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.autoscale(enable=True, tight=True)
        ax.set_title(ylabel + f" ($p_B = {pressure_uTorr:.2f}$ $\\mu$Torr)")

        if _data is not None:
            x_data = _data.axial_distance_m * xscalefactor
            y_data = _data.velocity_m_s
            y_data_mean = y_data.mean * yscalefactor
            y_data_std = y_data.std * yscalefactor
            ax.errorbar(
                x_data,
                y_data_mean,
                yerr=2 * y_data_std,
                fmt='o',
                color="black",
                label="Data",
                zorder=10,
                **data_kwargs,
            )

        sim_0 = sim[0][opcond].ion_velocity
        assert sim_0 is not None
        x_sim = sim_0.axial_distance_m * xscalefactor
        y_sim = np.array([_sim[opcond].ion_velocity.velocity_m_s.mean * yscalefactor for _sim in sim])

        qt = np.quantile(y_sim, q=QUANTILES, axis=0)
        _plot_quantiles(ax, x_sim, qt)
        medians[opcond] = qt[2]

        if map is not None and (map_data := map[opcond].ion_velocity) is not None:
            x_map = map_data.axial_distance_m * xscalefactor
            y_map = map_data.velocity_m_s.mean * yscalefactor
            ax.plot(x_map, y_map, label="Best sample", color='red')

        plot_name = f"{qty_name}_p={pressure_uTorr:05.2f}uTorr.png"
        _finalize_plot(fig, ax, ax_legend, out_path, plot_name)

    # Median predictions and best sample vs data on one plot
    fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 2)

    x_data_min = np.inf
    x_data_max = -np.inf

    for i, opcond in enumerate(opconds):
        _data = data[opcond].ion_velocity
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        color = colors(1 - (i + 0.5) / len(opconds))
        if _data is not None:
            x_data = _data.axial_distance_m * xscalefactor
            x_data_max = max(np.max(x_data), x_data_max)
            x_data_min = min(np.min(x_data), x_data_min)
            y_data = _data.velocity_m_s
            y_data_mean = y_data.mean * yscalefactor
            y_data_std = y_data.std * yscalefactor
            ax.errorbar(
                x_data,
                y_data_mean,
                yerr=2 * y_data_std,
                color=color,
                label=f"Data ($p_B = {pressure_uTorr:.2f}$ $\\mu$Torr)",
                zorder=1,
                fmt='--o',
                **data_kwargs,
            )

            sim_0 = sim[0][opcond].ion_velocity
            assert sim_0 is not None
            x_sim = sim_0.axial_distance_m * xscalefactor
            ax.plot(x_sim, medians[opcond], color=color, label="Median prediction", zorder=2)

    ax.set_xlim(x_data_min)
    plot_name = f"{qty_name}_allpressures.png"
    _finalize_plot(fig, ax, ax_legend, out_path, plot_name)


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

    qty_data_mean = np.array([x.mean for x in qty_data]) * scale
    qty_data_std = np.array([x.std for x in qty_data]) * scale

    mask_sim = np.array([getattr(x, quantity) is not None for x in sim[0].values()])
    pressure_sim = np.array([opcond.background_pressure_torr for opcond in sim[0].keys()])[mask_sim] * 1e6
    qty_sim = np.array(
        [[getattr(x, quantity).mean * scale for i, x in enumerate(_sim.values()) if mask_sim[i]] for _sim in sim]
    )

    if len(qty_data) == 0 and len(qty_sim) == 0:
        return

    sortperm_sim = np.argsort(pressure_sim)
    pressure_sim = pressure_sim[sortperm_sim]
    qty_sim = qty_sim[:, sortperm_sim]

    fig, (ax, ax_legend) = plt.subplots(1, 2, dpi=200, figsize=(10, 6), width_ratios=[3, 1])
    if lims is not None:
        ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(full_name)
    ax.set_xlim(0, np.max(pressure_sim) * 1.05)

    if len(qty_data) > 0:
        ax.errorbar(
            pressure_data,
            qty_data_mean,
            yerr=2 * qty_data_std,
            color="black",
            capsize=5,
            fmt='o',
            label="data",
            zorder=10,
        )

    if len(qty_sim) > 0:
        qt = np.quantile(qty_sim, QUANTILES, axis=0)
        _plot_quantiles(ax, pressure_sim, qt)

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
        amisc_path = mcmc_path / id

        # find pickle file in output dir
        pkl_file = "pem.pkl"
        for file in os.listdir(amisc_path):
            if file.endswith(".pkl"):
                pkl_file = file

        with open(amisc_path / pkl_file, "rb") as f:
            data.append(pickle.load(f))

    return data


def _plot_quantiles(ax, x, qt, label=True, fill=True, color=(0.3, 0.3, 0.3), zorder: int = 0):
    outer_args = {'linestyle': ':', 'color': color, 'zorder': 2}
    inner_args = {'linestyle': '-.', 'color': color, 'zorder': 2}
    inner_color = darken(color, 1.25)
    outer_color = darken(color, 2.0)

    ax.plot(
        x,
        qt[2],
        color=color,
        linestyle='--',
        linewidth=2,
        label="Median prediction" if label else "",
        zorder=zorder + 2,
    )
    if not np.all(qt[1] == qt[-2]):
        if fill:
            ax.fill_between(
                x, qt[1], qt[-2], facecolor=inner_color, label='50% CI' if label else "", zorder=zorder + 1, alpha=0.5
            )
        ax.plot(x, qt[1], **inner_args)
        ax.plot(x, qt[-2], **inner_args)
    if not np.all(qt[0] == qt[-1]):
        if fill:
            ax.fill_between(
                x, qt[0], qt[-1], facecolor=outer_color, label='90% CI' if label else "", zorder=zorder, alpha=0.5
            )
        ax.plot(x, qt[0], **outer_args)
        ax.plot(x, qt[-1], **outer_args)


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
    ax.hist(samples, nbins, color="lightgrey", density=True)


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

    parser.add_argument("--plot-corner", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--plot-bands", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    config_file = ""
    for file in os.listdir(args.path):
        if file.endswith(".yml") or file.endswith(".yaml"):
            config_file = file

    if config_file == "":
        raise FileNotFoundError(f"No YAML file found in {args.path}.")

    analyze_mcmc(Path(args.path), Path(config_file), args.datasets, corner=args.plot_corner, bands=args.plot_bands)
