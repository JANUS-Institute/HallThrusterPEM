import colorsys
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from amisc import System
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist, euclidean

import hallmd.data
import hallmd.utils
import pem_mcmc.io as io
from pem_mcmc.metrics import likelihood_and_distances
from pem_mcmc.types import Dataset

plt.rcParams.update(
    {
        "axes.formatter.use_mathtext": True,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.grid": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 18,
        "text.usetex": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

named_colors = {
    "black": "#000000ff",
    "white": "#ffffffff",
    "lightgrey": "#ccccccff",
    "grey": "#777777ff",
    "darkgrey": "#474747ff",
    "lightbrown": "#cf8577ff",
    "brown": "#9a4f3cff",
    "darkbrown": "#6c382dff",
    "lightred": "#fc4f30ff",
    "red": "#d62728ff",
    "darkred": "#8c1617ff",
    "lightorange": "#f8b722ff",
    "orange": "#dc8700ff",
    "darkorange": "#a3561cff",
    "lightgreen": "#bcbd22ff",
    "green": "#729f23ff",
    "darkgreen": "#547b25ff",
    "lightblue": "#17becfff",
    "blue": "#2a90c2ff",
    "darkblue": "#196599ff",
    "lightpurple": "#e377c2ff",
    "purple": "#9467bdff",
    "darkpurple": "#6a438bff",
}

data_kwargs = {'markersize': 3.5, 'capsize': 0}

legend_kwargs = {
    'fancybox': False,
    'framealpha': 1.0,
    'edgecolor': 'black',
    'borderpad': 0.3,
    'borderaxespad': 0.0,
    'handlelength': 1.0,
    'handletextpad': 0.4,
}

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _start_timer(msg: str) -> float:
    print(msg, "... ", end="", flush=True)
    return time.perf_counter_ns()


def _stop_timer(start_time: float):
    elapsed = time.perf_counter_ns() - start_time
    print(f"took {elapsed / 1e9:.2f} s.")


def analyze(
    path,
    config,
    datasets,
    plot_map=False,
    plot_corner=False,
    plot_bands=False,
    plot_traces=False,
    calc_metrics=False,
    proposal_cov=None,
    subsample=None,
    burn_fraction=0.0,
):
    mcmc_path = Path(path) / "mcmc"
    logfile = mcmc_path / "mcmc.csv"
    plot_path = Path(path) / "mcmc_analysis"
    os.makedirs(plot_path, exist_ok=True)
    print("Generating plots in", plot_path)

    analysis_start = time.time()

    system = System.load_from_file(Path(path) / config)
    device_name = system['Thruster'].model_kwargs['thruster']
    device = hallmd.utils.load_device(device_name)

    variables, samples, logposts, accepted, ids = io.read_output_file(logfile)

    num_burn = math.floor(burn_fraction * len(samples))

    samples_raw = np.array(samples)
    samples = np.array(samples)[num_burn:]
    logposts = np.array(logposts)
    map_ind = np.argmax(logposts)
    accepted = np.array(accepted)[num_burn:]

    num_accept = len(samples)
    tex_names = [system.inputs()[name].tex for name in variables]

    if num_accept > 10:
        start = _start_timer("Computing map, mean, median, and covariance")
        map = samples_raw[map_ind, :]
        mean, median = _mean_and_median(samples)
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

        _stop_timer(start)

        if plot_traces:
            start = _start_timer("Plotting traces")
            _plot_traces(tex_names, samples_raw, plot_path)
            _stop_timer(start)

        if plot_corner:
            try:
                start = _start_timer("Plotting corner plot")
                _plot_corner(samples, tex_names, plot_path, map=map, mean=mean, median=median)
                _stop_timer(start)
            except ValueError:
                # the corner plot sometimes fails early on if there aren't enough distinct samples
                pass

    start = _start_timer("Loading data")
    data = hallmd.data.load(hallmd.data.get_thruster(device_name).datasets_from_names(datasets))
    channel_length = device['geometry']['channel_length']
    map = _load_sim_results([ids[map_ind]], mcmc_path)[0]['output']
    _stop_timer(start)

    # TODO: factor this out into a dict or something
    if device_name == "SPT-100":
        # posterior lims
        thrust_lims = (50, 110)
        current_lims = (0, 7.5)
        vcc_lims = (28, 38)
        # prior lims
        # thrust_lims = (0, 150)
        # current_lims = (0, 65)
        # vcc_lims = (0, 60)
    else:
        thrust_lims = (0, 300)
        current_lims = (0, 100)
        vcc_lims = (0, 60)

    if subsample is not None and len(samples) > subsample:
        print(f"Subsampling {subsample} samples.")
        sample_inds = np.random.randint(low=num_burn, high=len(ids) - 1, size=subsample, dtype=int)
    else:
        sample_inds = np.arange(num_burn, len(ids))

    # Plot bands
    if plot_bands or calc_metrics:
        start = _start_timer("Loading results")
        results_all = _load_sim_results(np.array(ids)[sample_inds], mcmc_path)
        outputs = [res['output'] for res in results_all]
        _stop_timer(start)

        _map = map if plot_map else None

        metrics_out = None

        if calc_metrics:
            start = _start_timer("Calculating metrics")
            metrics = {k: [] for k in likelihood_and_distances(data, map)[1]}

            for output in outputs:
                _dist = likelihood_and_distances(data, output)[1]
                for k, (distance, _) in _dist.items():
                    metrics[k].append(distance)

            metrics_out = {k: {"mean": np.mean(v), "std": np.std(v)} for (k, v) in metrics.items()}

            _stop_timer(start)

        if plot_bands:
            start = _start_timer("Plotting thrust")
            thrust_median = _plot_global_quantity(
                data,
                outputs,
                plot_path,
                "thrust_N",
                "Thrust [mN]",
                map=_map,
                scale=1000,
                lims=thrust_lims,
                xscale="log",
            )

            plot_prediction_accuracy(data, outputs, plot_path, "thrust_N", "Thrust [mN]", scale=1000)

            _stop_timer(start)

            start = _start_timer("Plotting current")
            current_median = _plot_global_quantity(
                data,
                outputs,
                plot_path,
                "discharge_current_A",
                "Discharge current [A]",
                map=_map,
                lims=current_lims,
                xscale="log",
            )

            plot_prediction_accuracy(data, outputs, plot_path, "discharge_current_A", "Discharge current [A]")

            _stop_timer(start)

            start = _start_timer("Plotting cathode coupling voltage")
            vcc_median = _plot_global_quantity(
                data,
                outputs,
                plot_path,
                "cathode_coupling_voltage_V",
                "Cathode coupling voltage [V]",
                map=_map,
                lims=vcc_lims,
                xscale="log",
            )

            plot_prediction_accuracy(
                data, outputs, plot_path, "cathode_coupling_voltage_V", "Cathode coupling voltage [V]"
            )
            _stop_timer(start)

            start = _start_timer("Plotting ion velocity")
            uion_median = _plot_ion_vel(
                data,
                outputs,
                plot_path,
                map=_map,
                xlabel="Axial coordinate [channel lengths]",
                ylabel="Ion velocity [km/s]",
                xscalefactor=1 / channel_length,
                yscalefactor=1 / 1000,
                thruster=device_name,
            )
            _stop_timer(start)

            start = _start_timer("Plotting ion current density")
            jion_median = _plot_ion_cur(
                data,
                outputs,
                plot_path,
                xlabel="Angle [degrees]",
                ylabel="Ion current density [A/m$^2$]",
            )
            _stop_timer(start)

            if calc_metrics:
                assert thrust_median is not None
                assert current_median is not None
                assert vcc_median is not None
                assert uion_median is not None
                assert jion_median is not None

                median_dataset = {
                    opcond: hallmd.data.ThrusterData(
                        thrust_N=thrust_median[opcond],
                        discharge_current_A=current_median[opcond],
                        cathode_coupling_voltage_V=vcc_median[opcond],
                        ion_velocity=uion_median[opcond],
                        ion_current_sweeps=jion_median[opcond],
                    )
                    for opcond in data
                }

                assert metrics_out is not None

                metrics_median = {
                    k: distance for (k, (distance, _)) in likelihood_and_distances(data, median_dataset)[1].items()
                }
                for k, v in metrics_out.items():
                    v['median'] = metrics_median[k]

        if calc_metrics:
            assert metrics_out is not None

            with open(plot_path / "metrics.json", "w") as fd:
                json.dump(metrics_out, fd, indent=4)

    plt.close('all')

    print(f"Analysis finished in {time.time() - analysis_start:.2f} s.")


def _mean_and_median(X: np.ndarray, eps=1e-5):
    """
    Compute the empirical mean and median of a set of samples X.
    Based on StackOverflow answer here: https://stackoverflow.com/a/30305181/22854790
    """
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
    if ax_legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax_legend.legend(handles, labels, borderaxespad=0)
        ax_legend.axis("off")

    fig.tight_layout()

    if plot_name.endswith(".png"):
        _plot_name_noext = plot_name[0:-4]
    else:
        _plot_name_noext = plot_name

    for extension in ["png", "pdf"]:
        fig.savefig(out_path / (_plot_name_noext + "." + extension))
    plt.close(fig)


def clamp(x, x0, x1):
    return max(min(x, x1), x0)


def darken(color, factor):
    hsv = colorsys.rgb_to_hsv(float(color[0]), float(color[1]), float(color[2]))
    return colorsys.hsv_to_rgb(clamp(hsv[0], 0, 1), clamp(hsv[1] * 0.75, 0, 1), clamp(hsv[2] * factor, 0, 1))


def set_value(color, value):
    hsv = colorsys.rgb_to_hsv(float(color[0]), float(color[1]), float(color[2]))
    return colorsys.hsv_to_rgb(hsv[0], hsv[1], value)


def _plot_ion_cur(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    xlabel: str,
    ylabel: str,
):
    qty_name = "ion_current_density"
    out_path = plot_path / qty_name
    os.makedirs(out_path, exist_ok=True)

    # get the sweep radii
    pressures, _ = _extract_quantity(data, "ion_current_sweeps", sorted=True)
    _, sim_data_0 = _extract_quantity(sim[0], "ion_current_sweeps", sorted=True)
    sweep_radii = [x.radius_m for x in sim_data_0[0]]
    angles_sim_rad = sim_data_0[0][0].angles_rad
    angles_sim = angles_sim_rad * 180.0 / np.pi
    opconds = list(data.keys())

    colors = plt.get_cmap('turbo')
    xlims = (0, 90)
    yscale = 'log'

    data_kwargs = {'fmt': ':o', 'markersize': 4, 'alpha': 1}

    incr = 2

    # for each pressure, plot predictions at all radii
    jion_quantiles = {}
    medians = {}
    for i, opcond in enumerate(opconds):
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        _data = data[opcond].ion_current_sweeps
        if _data is None or len(data) == 0:
            medians[opcond] = None
            continue

        jion_quantiles[opcond] = []
        medians[opcond] = []

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

            jion_quantiles[opcond].append(qt)

            medians[opcond].append(
                hallmd.data.CurrentDensitySweep(
                    radius_m=sweep.radius_m,
                    angles_rad=angles_sim_rad,
                    current_density_A_m2=hallmd.data.Measurement(qt[2], np.full_like(qt[2], np.nan)),
                )
            )

        # Don't save plot if we only have one radius
        if len(sweep_radii) > 1:
            plot_name = f"{qty_name}_p={pressure_uTorr:05.2f}uTorr.png"
            _finalize_plot(fig, ax, ax_legend, out_path, plot_name)
        else:
            plt.close(fig)

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

            color = colors((j + 0.5) / len(pressures))
            sweep = _data[i]
            theta_deg = sweep.angles_rad * 180 / np.pi
            jion = sweep.current_density_A_m2

            for k, axis in enumerate([ax, ax_solo]):
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
                    axis.plot(angles_sim, qt[2], color=color, label="Median prediction" if j == 0 else "")
                else:
                    _plot_median_and_uncertainty(axis, angles_sim, qt)

            _finalize_plot(fig_solo, ax_solo, ax_legend_solo, subdir, plot_name)

        _finalize_plot(fig, ax, ax_legend, out_path, plot_name)

    return medians


def _plot_ion_vel(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    xlabel: str,
    ylabel: str,
    xscalefactor: float,
    yscalefactor: float,
    thruster: str,
    map: Dataset | None = None,
):
    qty_name = "ion_velocity"
    out_path = plot_path / qty_name
    os.makedirs(out_path, exist_ok=True)

    xlim = (0.0, 2.5)

    mask = np.array([getattr(x, qty_name) is not None for x in data.values()])
    opconds = [opcond for (i, opcond) in enumerate(data.keys()) if mask[i]]

    if len(opconds) == 0:
        return

    # Extract simulation coords and data
    medians = {}

    # Individual plots for each pressure
    for i, opcond in enumerate(data.keys()):
        sim_0 = sim[0][opcond].ion_velocity
        assert sim_0 is not None
        x_sim = sim_0.axial_distance_m * xscalefactor
        y_sim = np.array([_sim[opcond].ion_velocity.velocity_m_s.mean * yscalefactor for _sim in sim])

        qt = np.quantile(y_sim, q=QUANTILES, axis=0)

        # save medians for output
        medians[opcond] = hallmd.data.IonVelocityData(
            x_sim / xscalefactor, hallmd.data.Measurement(qt[2] / yscalefactor, np.full_like(qt[2], np.nan))
        )

        if not mask[i]:
            # Don't plot this operating condition if there's no data
            continue

        _data = data[opcond].ion_velocity
        pressure_uTorr = opcond.background_pressure_torr * 1e6

        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.autoscale(enable=True, tight=True)

        handles = []
        labels = []
        if _data is not None:
            x_data = _data.axial_distance_m * xscalefactor
            y_data = _data.velocity_m_s
            y_data_mean = y_data.mean * yscalefactor
            y_data_std = y_data.std * yscalefactor
            h_data = ax.errorbar(
                x_data,
                y_data_mean,
                yerr=2 * y_data_std,
                fmt='o',
                color="black",
                zorder=10,
                **data_kwargs,
            )
            handles.append(h_data)
            labels.append("Data")

        # Plot median and 90% credible intervals
        handles.append(_plot_median_and_uncertainty(ax, x_sim, qt))

        labels += ["Model"]

        # Plot best sample
        if map is not None and (map_data := map[opcond].ion_velocity) is not None:
            x_map = map_data.axial_distance_m * xscalefactor
            y_map = map_data.velocity_m_s.mean * yscalefactor
            ax.plot(x_map, y_map, label="Best sample", color='red')

        plot_name = f"{qty_name}_p={pressure_uTorr:05.2f}uTorr"
        ax.legend(handles, labels, loc='lower right', **legend_kwargs)

        _finalize_plot(fig, ax, None, out_path, plot_name)

    # ========================================================
    # Plot medians at selected pressures

    fig, ax = plt.subplots(dpi=200, figsize=(7, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)

    # Device-specific plotting settings
    match thruster:
        case "H9":
            inset_x = (0.93, 1.28)
            legend_loc = 'upper left'
        case _:
            inset_x = None
            legend_loc = 'lower right'

    axes = [ax]
    ax_inset = None
    if inset_x is not None:
        ax_inset = ax.inset_axes((0.6, 0.0, 0.4, 0.7), xlim=inset_x)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        axes.append(ax_inset)

    colors = ["red", "green", "blue"]
    if len(opconds) > 3:
        middle_index = round((len(opconds) - 1) / 2)
        inds = [0, middle_index, len(opconds) - 1]
    else:
        inds = list(range(len(opconds)))

    ymin = np.inf
    ymax = -np.inf

    handles = []
    labels = []

    for i, ind in enumerate(inds):
        opcond = opconds[ind]
        _data = data[opcond].ion_velocity
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        color = named_colors[colors[i]]
        if _data is not None:
            x_data = _data.axial_distance_m * xscalefactor
            y_data = _data.velocity_m_s
            y_data_mean = y_data.mean * yscalefactor
            y_data_std = y_data.std * yscalefactor

            if inset_x is not None:
                inset_mask = (x_data >= inset_x[0]) & (x_data <= inset_x[1])
                y_inset_mean = y_data_mean[inset_mask]
                y_inset_std = y_data_std[inset_mask]
                ymin = min(ymin, float(np.min(y_inset_mean - 2 * y_inset_std)))
                ymax = max(ymax, float(np.max(y_inset_mean + 2 * y_inset_std)))

            for iax, axis in enumerate(axes):
                handle = axis.errorbar(
                    x_data,
                    y_data_mean,
                    yerr=2 * y_data_std,
                    color=color,
                    fmt='o',
                    **data_kwargs,
                )

                if iax == 0:
                    labels.append(f"$p_B = {pressure_uTorr:.1f}$ $\\mu$Torr")
                    handles.append(handle)

            sim_0 = sim[0][opcond].ion_velocity
            assert sim_0 is not None
            x_sim = sim_0.axial_distance_m * xscalefactor
            for axis in axes:
                axis.plot(x_sim, medians[opcond].velocity_m_s.mean * yscalefactor, color=color, zorder=2, alpha=0.8)

    plot_name = f"{qty_name}_allpressures"

    if inset_x is not None:
        assert ax_inset is not None
        inset_width = 1.5
        ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=1, linewidth=inset_width)
        for direction in ['top', 'bottom', 'left', 'right']:
            ax_inset.spines[direction].set_linewidth(inset_width)
        pad = 0.02 * (ymax - ymin)
        ax_inset.set_ylim(ymin - pad, ymax + pad)

    handles.append(Line2D([0], [0], color="black", lw=2))
    labels.append("Median prediction")

    ax.legend(handles, labels, loc=legend_loc, **legend_kwargs)
    _finalize_plot(fig, ax, None, out_path, plot_name)
    return medians


def plot_prediction_accuracy(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    quantity: str,
    full_name: str,
    scale: float = 1,
    lims: Optional[tuple[float, float]] = None,
):
    lowercase_first_letter = lambda s: s[:1].casefold() + s[1:] if s else ""

    name_lowercase = lowercase_first_letter(full_name)
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.set_xlabel(f"Experimental {name_lowercase}")
    ax.set_ylabel(f"Predicted {name_lowercase}")

    x_mean = []
    x_err = []
    y_med = []
    y_err_lo = []
    y_err_hi = []

    for opcond in data.keys():
        qty_data = getattr(data[opcond], quantity)

        if qty_data is None:
            continue

        x_mean.append(qty_data.mean * scale)
        x_err.append(2 * qty_data.std * scale)

        qty_sim = np.array([getattr(s[opcond], quantity).mean * scale for s in sim])
        qt = np.quantile(qty_sim, QUANTILES, axis=0)
        lo, med, hi = qt[0], qt[2], qt[4]
        y_med.append(med)
        y_err_lo.append(med - lo)
        y_err_hi.append(hi - med)

    if len(x_mean) == 0:
        return

    x_mean = np.array(x_mean)
    x_err = np.array(x_err)
    y_med = np.array(y_med)
    y_err = [np.array(y_err_lo), np.array(y_err_hi)]

    color = "black"
    ax.errorbar(x_mean, y_med, xerr=x_err, yerr=y_err, fmt="none", color=color)
    ax.scatter(x_mean, y_med, color=color)

    # plot y = x line
    if lims is None:
        min_val = min(np.min(x_mean), np.min(y_med - y_err_lo))
        max_val = max(np.max(x_mean), np.max(y_med + y_err_hi))
        diff = max_val - min_val
        pad = 0.05 * diff
        max_val += pad
        min_val -= pad
    else:
        min_val, max_val = lims

    min_x = np.min(x_mean - x_err)
    max_x = np.max(x_mean + x_err)
    diff_x = max_x - min_x
    pad = 0.05 * diff_x
    max_x += pad
    min_x -= pad

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_val, max_val)

    line_x = np.array([min_x, max_x])
    ax.plot(line_x, line_x, color="red", zorder=4)
    fig.tight_layout()

    fig.savefig(plot_path / f"{quantity}_prediction.png")
    plt.close(fig)


def _plot_global_quantity(
    data: Dataset,
    sim: list[Dataset],
    plot_path: Path,
    quantity: str,
    full_name: str,
    map: Dataset | None = None,
    scale: float = 1,
    xscale: str = "linear",
    lims=None,
):
    xlabel = "Background pressure [Torr]"
    pressure_data, qty_data = _extract_quantity(data, quantity)
    pressure_data /= 1e6

    qty_data_mean = np.array([x.mean for x in qty_data]) * scale
    qty_data_std = np.array([x.std for x in qty_data]) * scale

    mask_sim = np.array([getattr(x, quantity) is not None for x in sim[0].values()])
    pressure_sim = np.array([opcond.background_pressure_torr for opcond in sim[0].keys()])[mask_sim]
    qty_sim = np.array(
        [[getattr(x, quantity).mean * scale for i, x in enumerate(_sim.values()) if mask_sim[i]] for _sim in sim]
    )

    if len(qty_data) == 0 and len(qty_sim) == 0:
        return

    sortperm_sim = np.argsort(pressure_sim)
    pressure_sim = pressure_sim[sortperm_sim]

    fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
    if lims is not None:
        ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(full_name)

    if xscale == "linear":
        ax.set_xlim(0, np.max(pressure_sim) * 1.05)
    else:
        ax.set_xscale(xscale)
        ax.set_xlim(left=np.min(pressure_sim) * 0.95, right=np.max(pressure_sim) * 1.05)

    handles = []
    labels = []

    if len(qty_data) > 0:
        handles.append(
            ax.errorbar(pressure_data, qty_data_mean, yerr=2 * qty_data_std, color="black", fmt='o', markersize=5)
        )
        labels.append("Data")

    median = None
    if len(qty_sim) > 0:
        qt = np.quantile(qty_sim, QUANTILES, axis=0)
        median = {o: hallmd.data.Measurement(q / scale, np.float64(np.nan)) for (o, q) in zip(data.keys(), qt[2])}

        # Sort quantiles by pressure for plotting
        qt = [q[sortperm_sim] for q in qt]
        handles.append(_plot_median_and_uncertainty(ax, pressure_sim, qt))
        labels.append("Model")

        if map is not None:
            pressure_map, qty_map = _extract_quantity(map, quantity)
            qty_map_mean = np.array([x.mean * scale for x in qty_map])
            ax.scatter(pressure_map, qty_map_mean, s=64, marker='x', color='red', label="Best sample", zorder=9)

    ax.legend(handles, labels, loc='upper left', **legend_kwargs)
    _finalize_plot(fig, ax, None, plot_path, f"{quantity}_pressure_bands")

    return median


def _load_sim_results(ids, mcmc_path: Path) -> list[dict]:
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


def _plot_median_and_uncertainty(ax, x, qt):
    h_uncertainty = ax.fill_between(x, qt[0], qt[-1], facecolor=named_colors["blue"], zorder=0, alpha=0.25)
    h_median = ax.plot(x, qt[2], color=named_colors["darkblue"])
    return h_uncertainty, h_median[0]


def _plot_traces(names, samples, dir: Path = Path(".")):
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


def _plot_corner(
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
