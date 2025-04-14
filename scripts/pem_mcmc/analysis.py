import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import amisc.distribution as distributions
import amisc.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from amisc import System, Variable
from matplotlib.axes import Axes
from scipy.spatial.distance import cdist, euclidean

import hallmd.data
import hallmd.utils
import pem_mcmc.io as io
from pem_mcmc.metrics import likelihood_and_distances
from pem_mcmc.types import Dataset

# Common styling for all plots
RCPARAMS = {
    "axes.formatter.use_mathtext": True,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.grid": True,
    "errorbar.capsize": 0.0,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 18,
    "text.usetex": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.0,
    "legend.edgecolor": 'black',
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.4,
}
plt.rcParams.update(RCPARAMS)

# Pre-selected named colors that look good together
COLORS = {
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
    "darkgreen": "#3e5d19ff",
    "lightblue": "#c7ecffff",
    "blue": "#2a90c2ff",
    "darkblue": "#196599ff",
    "lightpurple": "#e377c2ff",
    "purple": "#9467bdff",
    "darkpurple": "#6a438bff",
}

# axis limits for different thrusters and quantities
AXIS_LIMITS = {
    "SPT-100": {
        "prior": {
            "thrust": (0.0, 150.0),
            "current": (0.0, 65.0),
            "vcc": (0.0, 60.0),
        },
        "posterior": {
            "thrust": (60.0, 90.0),
            "current": (3.0, 6.0),
            "vcc": (28.0, 38.0),
        },
        "test": {
            "thrust": (0.0, 130.0),
            "current": (0.0, 80.0),
            "vcc": (0.0, 60.0),
        },
    },
    "H9": {
        "prior": {
            "thrust": (0.0, 300.0),
            "current": (0.0, 100.0),
            "vcc": (0.0, 60.0),
        },
        "posterior": {
            "thrust": (150.0, 300.0),
            "current": (5.0, 20.0),
            "vcc": (0.0, 20.0),
        },
        "test": {
            "thrust": (0.0, 300.0),
            "current": (0.0, 100.0),
            "vcc": (0.0, 60.0),
        },
    },
}

data_kwargs = {'markersize': 3.5}

# Sort order for pemv1 variables.
# For other systems, we do not try and sort the variables
sort_order = [
    "T_e",
    "V_vac",
    "P_T",
    "Pstar",
    "anom_scale",
    "anom_barrier_scale",
    "anom_center",
    "anom_width",
    "anom_shift_length",
    "u_n",
    "c_w",
    "f_n",
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
]

# Quantiles for computing error bars
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def try_sort_variables(vars: dict[Variable, np.ndarray], sort_order: list[Variable]) -> dict[Variable, np.ndarray]:
    # Check that the keys are the same
    input_vars = set(vars.keys())
    if input_vars != set(sort_order):
        return vars

    return {var: vars[var] for var in sort_order}


def pad_limits(lims: tuple[float, float], pad: float) -> tuple[float, float]:
    min_val, max_val = lims
    pad_amt = 0.5 * pad * (max_val - min_val)
    return (min_val - pad_amt, max_val + pad_amt)


def _start_timer(msg: str) -> float:
    print(msg, "... ", end="", flush=True)
    return time.perf_counter_ns()


def _stop_timer(start_time: float):
    elapsed = time.perf_counter_ns() - start_time
    print(f"took {elapsed / 1e9:.2f} s.")


def analyze(
    amisc_dir,
    datasets,
    plot_map=False,
    plot_corner=False,
    plot_bands=False,
    plot_traces=False,
    calc_metrics=False,
    proposal_cov=None,
    subsample=None,
    burn_fraction=0.0,
    limits="posterior",
):
    dir = Path(amisc_dir)
    mcmc_path = dir / "mcmc"
    logfile = mcmc_path / "mcmc.csv"
    plot_path = dir / "mcmc_analysis"
    os.makedirs(plot_path, exist_ok=True)
    print("Generating plots in", plot_path)

    analysis_start = time.time()

    system = io.load_system(dir)
    device_name = system['Thruster'].model_kwargs['thruster']
    device = hallmd.utils.load_device(device_name)

    variables, samples, logposts, accepted, ids = io.read_output_file(logfile)
    num_burn = math.floor(burn_fraction * len(samples))

    samples_raw = samples
    samples = samples[num_burn:]
    map_ind = np.argmax(logposts)
    accepted = accepted[num_burn:]

    num_accept = len(samples)
    tex_names = [system.inputs()[name].tex for name in variables]

    sorted_vars = [system.inputs()[var] for var in sort_order]
    variable_dict = try_sort_variables(
        {system.inputs()[name]: samples[:, i] for (i, name) in enumerate(variables)}, sorted_vars
    )

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
            start = _start_timer("Plotting corner plot")
            # the corner plot sometimes fails early on if there aren't enough distinct samples
            try:
                make_componentwise_cornerplots(variable_dict, system, plot_path)
            except ValueError:
                pass
            _stop_timer(start)

        # Compute parameter statistics and dump to a LaTeX table
        compute_sample_statistics(variable_dict, plot_path)

    if plot_bands or calc_metrics:
        start = _start_timer("Loading data")
        data = hallmd.data.load(hallmd.data.get_thruster(device_name).datasets_from_names(datasets))
        channel_length = device['geometry']['channel_length']
        map = _load_sim_results([ids[map_ind]], mcmc_path)

        if map:
            map = map[0]['output']
        else:
            map = None

        if device_name.casefold() and map is not None == "h9":
            assert isinstance(map, dict)
            map = _merge_opconds([map], data)[0]

        _stop_timer(start)

        if subsample is not None and len(samples) > subsample:
            print(f"Subsampling {subsample} samples.")
            sample_inds = np.random.randint(low=num_burn, high=len(ids) - 1, size=subsample, dtype=int)
        else:
            sample_inds = np.arange(num_burn, len(ids))

        # Plot bands
        start = _start_timer("Loading results")
        results_all = _load_sim_results(np.array(ids)[sample_inds], mcmc_path)

        outputs = [res['output'] for res in results_all]

        if device_name.casefold() == "h9":
            outputs = _merge_opconds(outputs, data)

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
            thrust_lims = AXIS_LIMITS[device_name][limits]["thrust"]
            current_lims = AXIS_LIMITS[device_name][limits]["current"]
            vcc_lims = AXIS_LIMITS[device_name][limits]["vcc"]

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

            _plot_prediction_accuracy(data, outputs, plot_path, "thrust_N", "Thrust [mN]", scale=1000, lims=thrust_lims)

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

            _plot_prediction_accuracy(
                data, outputs, plot_path, "discharge_current_A", "Discharge current [A]", lims=current_lims
            )

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

            _plot_prediction_accuracy(
                data, outputs, plot_path, "cathode_coupling_voltage_V", "Cathode coupling voltage [V]", lims=vcc_lims
            )
            _stop_timer(start)

            start = _start_timer("Plotting ion velocity")
            uion_median, uion_pressures = _plot_ion_vel(
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

            start = _start_timer("Plotting anom. transport")
            plot_anom(variable_dict, uion_pressures, plot_path)

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
                median_dataset = {
                    opcond: hallmd.data.ThrusterData(
                        thrust_N=thrust_median[opcond] if thrust_median is not None else None,
                        discharge_current_A=current_median[opcond] if current_median is not None else None,
                        cathode_coupling_voltage_V=vcc_median[opcond] if vcc_median is not None else None,
                        ion_velocity=uion_median[opcond] if uion_median is not None else None,
                        ion_current_sweeps=jion_median[opcond] if jion_median is not None else None,
                    )
                    for opcond in data
                }

                assert metrics_out is not None

                metrics_median = {
                    k: distance for (k, (distance, _)) in likelihood_and_distances(data, median_dataset)[1].items()
                }
                for k, v in metrics_median.items():
                    metrics_out[k]['median'] = v

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


def save_figure(fig, out_path: os.PathLike, plot_name: str, tight_layout: bool = True):
    if tight_layout:
        fig.tight_layout()

    if plot_name.endswith(".png"):
        _plot_name_noext = plot_name[0:-4]
    else:
        _plot_name_noext = plot_name

    for extension in ["png", "pdf"]:
        fig.savefig(Path(out_path) / (_plot_name_noext + "." + extension))
    plt.close(fig)


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
    _, sim_data_0 = _extract_quantity(sim[0], "ion_current_sweeps", sorted=True)
    sweep_radii = [x.radius_m for x in sim_data_0[0]]
    angles_sim_rad = sim_data_0[0][0].angles_rad
    angles_sim = angles_sim_rad * 180.0 / np.pi
    opconds = list(data.keys())

    colors = plt.get_cmap('turbo')
    xlims = (0, 90)
    yscale = 'log'

    max_points = 40

    # for each pressure, plot predictions at all radii
    jion_quantiles = {}
    medians = {}
    colors = ["red", "lightorange", "green", "darkblue"]
    markerstyles = ["o", "v", "^", ">"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    for i, opcond in enumerate(opconds):
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        _data = data[opcond].ion_current_sweeps
        if _data is None or len(data) == 0:
            medians[opcond] = None
            continue

        jion_quantiles[opcond] = []
        medians[opcond] = []

        fig, ax = plt.subplots(dpi=200, figsize=(7, 6))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlims)

        # plot data for this operating condition at each radius
        handles_data = []
        labels_data = []
        for j, sweep in enumerate(_data):
            color = COLORS[colors[j]]

            theta_deg = sweep.angles_rad * 180 / np.pi
            jion = sweep.current_density_A_m2
            incr = max(1, math.floor(len(theta_deg) / max_points))
            h_data = ax.errorbar(
                theta_deg[::incr],
                jion.mean[::incr],
                yerr=2 * jion.std[::incr],
                color=color,
                fmt=markerstyles[j],
                markersize=5,
            )

            jion_sim = [_sim[opcond].ion_current_sweeps[j].current_density_A_m2.mean for _sim in sim]
            qt = np.quantile(jion_sim, q=QUANTILES, axis=0)

            # Plot median prediction
            h_sim = ax.plot(angles_sim, qt[2], color=color, linestyle=linestyles[j])[0]

            handles_data.append((h_sim, h_data))
            labels_data.append(f"$r =$ {sweep.radius_m:.2f} m")

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
            ax.legend(handles_data, labels_data, loc='upper right')
            save_figure(fig, out_path, plot_name)
        else:
            plt.close(fig)

    colors = ["red", "green", "blue"]
    linestyles = ["solid", "dashed", "dashdot"]
    markerstyles = ["o", "v", "^"]
    opconds_ji = [o for o in opconds if data[o].ion_current_sweeps is not None]
    if len(opconds_ji) > 3:
        middle_index = round((len(opconds_ji) - 1) / 2)
        inds = [0, middle_index, len(opconds_ji) - 1]
    else:
        inds = list(range(len(opconds_ji)))

    # for each radius, plot predictions at up to three pressures
    for i, sweep_radius in enumerate(sweep_radii):
        fig, ax = plt.subplots(dpi=200, figsize=(7, 6))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlims)
        plot_name = f"{qty_name}_r={sweep_radius:.2f}m.png"

        # plot data for this operating condition at each pressure
        handles_data = []
        labels_data = []
        for j, opcond_ind in enumerate(inds):
            opcond = opconds_ji[opcond_ind]
            color = COLORS[colors[j]]
            pressure_uTorr = opcond.background_pressure_torr * 1e6
            _data = data[opcond].ion_current_sweeps
            if _data is None or len(data) == 0:
                continue

            # make dedicated plot for each opcond
            fig_solo, ax_solo = plt.subplots(dpi=200, figsize=(7, 6))
            ax_solo.set_yscale(yscale)
            ax_solo.set_xlabel(xlabel)
            ax_solo.set_ylabel(ylabel)
            ax_solo.set_xlim(xlims)
            pressure_str = f"${pressure_uTorr:.2f}$ $\\mu$Torr"
            subdir = out_path / f"p={pressure_uTorr:05.2f}uTorr"
            os.makedirs(subdir, exist_ok=True)

            sweep = _data[i]
            theta_deg = sweep.angles_rad * 180 / np.pi
            jion = sweep.current_density_A_m2
            incr = max(1, math.floor(len(theta_deg) / max_points))

            for k, axis in enumerate([ax, ax_solo]):
                h_data = axis.errorbar(
                    theta_deg[::incr],
                    jion.mean[::incr],
                    yerr=2 * jion.std[::incr],
                    color=color if k == 0 else 'black',
                    fmt=markerstyles[j] if k == 0 else 'o',
                    markersize=4.5,
                )
                qt = jion_quantiles[opcond][i]
                if k == 0:
                    h_sim = axis.plot(angles_sim, qt[2], color=color, linestyle=linestyles[j])[0]
                    handles_data.append((h_sim, h_data))
                    labels_data.append(f"{pressure_str}")
                else:
                    h_model = plot_median_and_uncertainty(axis, angles_sim, qt)
                    ax_solo.legend([h_data, h_model], ["Data", "Model (median + 90\\% CI)"], loc='upper right')

            save_figure(fig_solo, subdir, plot_name)

        ax.legend(handles_data, labels_data, loc='upper right')
        save_figure(fig, out_path, plot_name)

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
) -> tuple[dict[hallmd.data.OperatingCondition, hallmd.data.IonVelocityData], list[float]]:
    qty_name = "ion_velocity"
    out_path = plot_path / qty_name
    os.makedirs(out_path, exist_ok=True)

    xlim = (0.0, 2.5)

    mask = np.array([getattr(x, qty_name) is not None for x in data.values()])
    opconds = [opcond for (i, opcond) in enumerate(data.keys()) if mask[i]]

    # Extract simulation coords and data
    medians = {}

    if len(opconds) == 0:
        return {}, []

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

        fig, ax = plt.subplots(dpi=200, figsize=(7, 6))
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
        handles.append(plot_median_and_uncertainty(ax, x_sim, qt))

        labels += ["Model (median and 90\\% CI)"]

        # Plot best sample
        if map is not None and (map_data := map[opcond].ion_velocity) is not None:
            x_map = map_data.axial_distance_m * xscalefactor
            y_map = map_data.velocity_m_s.mean * yscalefactor
            ax.plot(x_map, y_map, label="Best sample", color='red')

        plot_name = f"{qty_name}_p={pressure_uTorr:05.2f}uTorr"
        ax.legend(handles, labels, loc='lower right')

        save_figure(fig, out_path, plot_name)

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
    linestyles = ["solid", "dashed", "dashdot"]
    markerstyles = ["o", "v", "^"]
    if len(opconds) > 3:
        middle_index = round((len(opconds) - 1) / 2)
        inds = [0, middle_index, len(opconds) - 1]
    else:
        inds = list(range(len(opconds)))

    pressures = [opconds[ind].background_pressure_torr for ind in inds]

    ymin = np.inf
    ymax = -np.inf

    handles = []
    labels = []

    for i, ind in enumerate(inds):
        opcond = opconds[ind]
        _data = data[opcond].ion_velocity
        pressure_uTorr = opcond.background_pressure_torr * 1e6
        color = COLORS[colors[i]]
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
                    fmt=markerstyles[i],
                    markersize=4.5,
                )

                if iax == 0:
                    labels.append(f"${pressure_uTorr}$ $\\mu$Torr")
                    handles.append(handle)

            sim_0 = sim[0][opcond].ion_velocity
            assert sim_0 is not None
            x_sim = sim_0.axial_distance_m * xscalefactor
            for iax, axis in enumerate(axes):
                handle = axis.plot(
                    x_sim,
                    medians[opcond].velocity_m_s.mean * yscalefactor,
                    color=color,
                    zorder=2,
                    alpha=0.8,
                    linestyle=linestyles[i],
                )[0]

                if iax == 0:
                    handles[i] = (handle, handles[i])

    plot_name = f"{qty_name}_allpressures"

    if inset_x is not None:
        assert ax_inset is not None
        inset_width = 1.5
        ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=1, linewidth=inset_width)
        for direction in ['top', 'bottom', 'left', 'right']:
            ax_inset.spines[direction].set_linewidth(inset_width)
        pad = 0.02 * (ymax - ymin)
        ax_inset.set_ylim(ymin - pad, ymax + pad)

    # handles.append(Line2D([0], [0], color="black", lw=2))
    # labels.append("Median prediction")

    ax.legend(handles, labels, loc=legend_loc)
    save_figure(fig, out_path, plot_name)
    return medians, pressures


def _plot_prediction_accuracy(
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

    # build matrix to save to file so we can plot multiple runs on the sample plot
    out_header = "data mean,data err,sim 0.5 percentile,sim 0.05 percentile, sim 0.95 percentile"
    out_data = np.vstack([x_mean, x_err, y_med, y_err_lo, y_err_hi]).T
    np.savetxt(plot_path / f"{quantity}.csv", out_data, delimiter=",", header=out_header)

    color = "black"
    ax.errorbar(x_mean, y_med, xerr=x_err, yerr=y_err, fmt="none", color=color, markersize=3, alpha=0.5, zorder=4)
    ax.scatter(x_mean, y_med, color=color, linewidth=0, s=10, zorder=5)

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

    save_figure(fig, plot_path, f"{quantity}_prediction")
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

    fig, ax = plt.subplots(dpi=200, figsize=(7, 6))
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
        handles.append(plot_median_and_uncertainty(ax, pressure_sim, qt))
        labels.append("Model")

        if map is not None:
            pressure_map, qty_map = _extract_quantity(map, quantity)
            qty_map_mean = np.array([x.mean * scale for x in qty_map])
            ax.scatter(pressure_map, qty_map_mean, s=64, marker='x', color='red', label="Best sample", zorder=9)

    ax.legend(handles, labels, loc='upper left')
    save_figure(fig, plot_path, f"{quantity}_pressure_bands")

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

        if not os.path.exists(amisc_path / pkl_file):
            continue

        with open(amisc_path / pkl_file, "rb") as f:
            ds = pickle.load(f)
            if ds['output'] is None:
                print(f"{amisc_path=}, {id=}")
            data.append(ds)

    return data


def _merge_opconds(sim: list[dict], data: Dataset) -> list[dict]:
    """
    We changed the input mass flow rate for the H9 to fix a bug, but that means the sim mass flow rates no longer
    match up with the data. This function fixes that by finding the conditions with the same pressures as the data.
    """

    sim_to_data = {}

    for sim_opcond in sim[0].keys():
        for data_opcond in data.keys():
            if sim_opcond.background_pressure_torr == data_opcond.background_pressure_torr:
                sim_to_data[sim_opcond] = data_opcond
                break

    return [{sim_to_data[k]: v for (k, v) in s.items()} for s in sim]


def plot_median_and_uncertainty(
    ax, x, qt, alpha=0.25, lightcolor=COLORS["blue"], darkcolor=COLORS["darkblue"], linestyle='-'
):
    h_uncertainty = ax.fill_between(x, qt[0], qt[-1], facecolor=lightcolor, zorder=0, alpha=alpha)
    h_median = ax.plot(x, qt[2], color=darkcolor, linestyle=linestyle)
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


def make_componentwise_cornerplots(samples: dict[Variable, np.ndarray], system: System, output_dir: Path):
    component_vars = split_vars_by_components(samples, system)

    # Plot individual corner plots per-component
    for comp_name, var_dict in component_vars.items():
        make_cornerplot(var_dict, output_dir, comp_name.casefold())


def make_cornerplot(samples: dict[Variable, np.ndarray], output_dir: Path, plot_name: str = ""):
    num_vars = len(samples.keys())
    subfig_size = 2.0
    rows = num_vars

    fig, axes = plt.subplots(
        rows,
        rows,
        dpi=200,
        figsize=(rows * subfig_size, rows * subfig_size),
        layout="constrained",
    )

    limits = [pad_limits((np.min(samples), np.max(samples)), 0.2) for samples in samples.values()]

    for irow, rowvar in enumerate(samples.keys()):
        for icol, colvar in enumerate(samples.keys()):
            axis = axes[irow, icol]
            if irow < icol:
                axis.set_visible(False)
                continue

            axis.grid(False)

            if irow == rows - 1:
                axis.set_xlabel(colvar.tex)
            else:
                axis.set_xticklabels([])

            if icol == 0 and irow > 0:
                axis.set_ylabel(rowvar.tex)
            else:
                axis.set_yticklabels([])

            axis.set_xlim(limits[icol])

            if irow == icol:
                # 1D histogram - kernel density estimate
                _axis_hist_1D(axis, samples[colvar], limits[icol])
                axis.set_yticks([])
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.set_ylim(bottom=0)
                continue
            else:
                axis.hexbin(samples[colvar], samples[rowvar], gridsize=12, mincnt=1)
                axis.set_ylim(limits[irow])

    if plot_name:
        plot_name = "corner_" + plot_name
    else:
        plot_name = "corner"

    save_figure(fig, output_dir, plot_name, tight_layout=False)


def _axis_hist_1D(axis: Axes, samples: np.ndarray, lims: tuple[float, float], N: int = 100):
    kernel = scipy.stats.gaussian_kde(samples)
    xvals = np.linspace(lims[0], lims[1], N)
    probs = kernel(xvals)
    axis.fill_between(xvals, np.zeros(N), probs, color=COLORS["lightblue"])
    axis.plot(xvals, probs, linewidth=3, color=COLORS["blue"])
    axis.set_xlim(lims)


def split_vars_by_components(vars: dict[Variable, np.ndarray], system: System) -> dict[str, dict[Variable, np.ndarray]]:
    return {
        comp.name: {
            v: samples
            for (v, samples) in vars.items()
            if v in comp.inputs and not (comp.name == "Thruster" and v.name == "T_e")
        }
        for comp in system.components
    }


def compute_var_statistics(var: Variable, samples: np.ndarray):
    quantiles = np.quantile(samples, QUANTILES)

    transform_str = "None" if var.norm is None else f"{var.norm[0]}"

    return {
        "dist": f"{var.distribution}",
        "transform": transform_str,
        "min": np.min(samples),
        "5%": quantiles[0],
        "50%": quantiles[2],
        "95%": quantiles[4],
        "max": np.max(samples),
        "std": np.std(samples),
    }


def rounded_number(x):
    if type(x) is str:
        if x == '':
            x = 0
    f = float(x)
    if f.is_integer():
        return int(f), True

    if abs(f - round(f)) < 0.01:
        return rounded_number(round(f))[0], True

    if abs(f - round(f, ndigits=1)) < 0.001:
        return round(f, ndigits=1), True
    return f, False


def format_number(x, check_pi=False):
    f, was_rounded = rounded_number(x)
    if was_rounded:
        return f"{f}"

    # check for multiples of pi
    if check_pi:
        if f > np.pi:
            div = f / np.pi
            if abs(div - round(div)) < 0.01:
                return f"\\{round(div)}\\pi"

        if f < np.pi:
            # check for fractions of pi
            fractions = [2, 3, 4, 6]
            for frac in fractions:
                if abs(np.pi / frac - f) < 0.01:
                    return f"\\pi/{frac}"

    return f"{f:.2f}"


def compute_sample_statistics(samples: dict[Variable, np.ndarray], output_dir: Path):
    output = {var.name: compute_var_statistics(var, _samples) for (var, _samples) in samples.items()}

    with open(output_dir / "variable_stats.json", "w") as fd:
        json.dump(output, fd)

    cols = 8
    col_str = " ".join("l" * cols)
    indent = "    "

    def nth(n):
        return f"${n}^{{\\text{{th}}}}$ pctile"

    # Create latex table
    latex_header = f"""\\begin{{tabular}}{{{col_str}}}
{indent}\\hline
{indent}\\multicolumn{{2}}{{l}}{{}} & \\multicolumn{{{cols - 2}}}{{l}}{{Posterior}} \\\\ \\cline{{3-{cols}}}
{indent}Variable & Prior & Min & {nth(5)} & {nth(50)} & {nth(95)} & Max & Std dev \\\\ \\hline"""

    notes = [
        "Variables with the ($10^x$) notation indicate a log-uniform distribution.",
        "The ($x$) notation indicates that the variable has been normalized by $x$.",
    ]

    note_str = "\\\\\n".join([indent + f"\\multicolumn{{{cols}}}{{l}}{{{note}}}" for note in notes])

    latex_footer = f"{indent}\\hline\n{note_str}\n\\end{{tabular}}"

    rows = []

    for var, stats in zip(samples.keys(), output.values()):
        var_str = f"{var.tex}"

        dist = var.distribution

        if isinstance(dist, distributions.LogUniform):
            var_str += " ($10^{x}$)"

        if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
            lo, hi = dist.dist_args
            lo = format_number(var.normalize(lo), check_pi=True)
            hi = format_number(var.normalize(hi), check_pi=True)
            dist_str = f"$\\mathcal{{U}}({lo}, {hi})$"
        else:
            dist_str = ""

        if var.norm is not None and isinstance(var.norm[0], transform.Linear):
            exponent = format_number(np.log10(var.norm[0].transform_args[0]))
            var_str += f" (\\num{{e{exponent}}})"

        min_str = format_number(stats['min'])
        max_str = format_number(stats['max'])
        std_str = format_number(stats['std'])
        pct_5 = format_number(stats['5%'])
        pct_50 = format_number(stats['50%'])
        pct_95 = format_number(stats['95%'])

        cols = [var_str, dist_str, min_str, pct_5, pct_50, pct_95, max_str, std_str]
        row_str = " & ".join(cols) + "\\\\"

        rows.append(indent + row_str)

    with open(output_dir / "variable_table.tex", "w") as fd:
        print(latex_header, file=fd)
        print("\n".join(rows), file=fd)
        print(latex_footer, file=fd)


def simple_pressure_shift(pressure, shift_length, slope=2.0, midpoint=25e-6):
    ratio = pressure / midpoint
    return -shift_length * (1 / (1 + np.exp(-slope * (ratio - 1))) - 1 / (1 + np.exp(slope)))


def scaled_gaussian_bohm(coords, anom_scale, anom_barrier_scale, anom_center, anom_width, z_shift=0.0):
    return anom_scale * (1 - anom_barrier_scale * np.exp(-0.5 * ((coords - z_shift - anom_center) / (anom_width)) ** 2))


def plot_anom(samples: dict[Variable, np.ndarray], pressures: list[float], output_dir: Path):
    var_name_dict = {v.name: s for (v, s) in samples.items()}
    anom_scale = var_name_dict["anom_scale"]
    anom_barrier_scale = var_name_dict["anom_barrier_scale"]
    anom_center = var_name_dict["anom_center"]
    anom_width = var_name_dict["anom_width"]
    shift_length = var_name_dict["anom_shift_length"]

    lightcolors = ["lightred", "lightgreen", "lightblue"]
    darkcolors = ["red", "green", "darkblue"]
    linestyles = ['-', '--', '-.']

    if len(pressures) > 3:
        p = pressures[:3]
    else:
        p = pressures

    fig, axis = plt.subplots(figsize=(5, 4.5), dpi=200)
    left = 0
    right = 3
    coords = np.linspace(left, right, 100)
    axis.set_xlim(left, right)
    axis.set_xlabel("Axial coordinate [channel lengths]")
    axis.set_ylabel("$\\nu_{anom} / (\\omega_{ce} / 16)$")

    handles = []
    labels = []

    for pressure, light, dark, style in zip(p, lightcolors, darkcolors, linestyles):
        anoms = []

        for i, (alpha, beta, z, L, dz) in enumerate(
            zip(anom_scale, anom_barrier_scale, anom_center, anom_width, shift_length)
        ):
            z_shift = simple_pressure_shift(pressure, dz)
            anoms.append(scaled_gaussian_bohm(coords, alpha, beta, z, L, z_shift=z_shift) * 16)

        quantiles = np.quantile(np.array(anoms), QUANTILES, axis=0)

        handles.append(
            plot_median_and_uncertainty(
                axis,
                coords,
                quantiles,
                alpha=0.5,
                lightcolor=COLORS[light],
                darkcolor=COLORS[dark],
                linestyle=style,
            )
        )

        labels.append(f"{pressure / 1e-6:.1f} $\\mu$Torr")

    axis.legend(handles, labels)
    axis.set_ylim(bottom=0, top=2)
    save_figure(fig, output_dir, "anom")
