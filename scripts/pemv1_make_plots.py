import argparse
import json
import logging
import os
import shutil
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pem_mcmc.analysis as mcmc_analysis
import pem_mcmc.io as io
from matplotlib.axes import Axes
from pem_mcmc.analysis import COLORS, RCPARAMS, GlobalQuantityData

import hallmd.data

plt.rcParams.update(RCPARAMS)
THRUSTERS = ["h9", "spt-100"]

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=".")
parser.add_argument("-p", "--plots", action='append', type=lambda s: s.split(','))
parser.add_argument("-l", "--loglevel", type=int, default=logger.level)
parser.add_argument("--reanalyze", action="store_true")
parser.add_argument("--range", type=str, choices=["prior", "posterior", "test"], default="posterior")

parser.add_argument("--prior-train", type=str)
parser.add_argument("--prior-test", type=str)
parser.add_argument("--post-train-epistemic", type=str)
parser.add_argument("--post-train-aleatoric", type=str)
parser.add_argument("--post-test", type=str)
parser.add_argument("--thruster", type=str, choices=THRUSTERS)

QUANTITY_NAMES = {
    "thrust_N": "thrust [mN]",
    "discharge_current_A": "discharge current [A]",
    "cathode_coupling_voltage_V": "cathode coupling voltage [V]",
}


def get_thruster_name(args: argparse.Namespace, logger: Logger) -> str | None:
    dirs = [args.prior_train, args.prior_test, args.post_train_epistemic, args.post_train_aleatoric, args.post_test]

    systems = [io.load_system(Path(d)) for d in dirs if d is not None]

    if systems:
        thrusters = [system['Thruster'].model_kwargs['thruster'].casefold() for system in systems]
        directory_thrusters = set(thrusters)
        if len(directory_thrusters) > 1:
            logger.error(f'Input directories contain data from multiple thrusters: {directory_thrusters}. Exiting.')
            return None
        else:
            directory_thruster = list(directory_thrusters)[0]
    else:
        directory_thruster = None

    if args.thruster is None:
        return directory_thruster
    elif directory_thruster is None:
        return args.thruster

    if args.thruster != directory_thruster:
        logger.error(
            f"Mismatch between inferred ({directory_thruster}) and specified ({args.thruster}) thrusters. Exiting."
        )
        return None

    return directory_thruster


ALL_PLOTS = {'anom_diagram', 'mdot', 'predict', "metrics"}


def main(args: argparse.Namespace):
    output_dir = Path(args.output)
    if output_dir.is_file():
        raise ValueError(f"Provided output directory `{output_dir}`is a file!")

    output_dir.mkdir(exist_ok=True)

    # Flatten plot type list
    plot_types = set([p for ps in args.plots for p in ps])
    if 'all' in plot_types:
        plot_types.remove('all')
        plot_types = plot_types.union(ALL_PLOTS)

    # Remove invalid plots
    invalid_plots = plot_types.difference(ALL_PLOTS)
    if invalid_plots:
        logger.warning(f"Ignoring unrecognized plot types {invalid_plots}.")

    plot_types = plot_types.intersection(ALL_PLOTS)
    logger.info(f"{plot_types=}")

    # General plots that don't rely on results
    if "anom_diagram" in plot_types:
        paths = _make_anom_diagram(output_dir, logger)
        logger.info(f"Anom diagram plot saved to {[p.__str__() for p in paths]}")
        plot_types.remove("anom_diagram")

    # Get thruster name
    thruster = get_thruster_name(args, logger)
    if thruster is None:
        logger.error(f"Thruster type required for plot types {plot_types}. Exiting.")
        return
    else:
        thruster_dir = output_dir / thruster

    if "mdot" in plot_types:
        paths = _make_mdot_plot(thruster, thruster_dir, logger)
        if paths:
            logger.info(f"Mass flow rate plot saved to {[p.__str__() for p in paths]}")
        plot_types.remove("mdot")

    # Plots that require results
    prior_test = Path(p) if (p := args.prior_test) else None
    post_test = Path(p) if (p := args.post_test) else None
    prior_train = Path(p) if (p := args.prior_train) else None
    post_train_aleatoric = Path(p) if (p := args.post_train_aleatoric) else None
    post_train_epistemic = Path(p) if (p := args.post_train_epistemic) else None

    if thruster == "h9":
        training_datasets = ["um2024"]
        test_datasets = ["gt2024"]
    elif thruster == "spt-100":
        training_datasets = ["diamant2014", "macdonald2019"]
        test_datasets = ["sankovic1993"]
    else:
        assert False

    if args.reanalyze:
        mcmc_analysis.analyze(
            amisc_dir=post_train_epistemic,
            datasets=training_datasets,
            # plot_corner=True,
            # plot_bands=True,
            # calc_metrics=True,
            subsample=1000,
            burn_fraction=0.7,
            output_dir=thruster_dir / "post_train_epistemic",
            secondary_simulation=post_train_aleatoric,
            limits=args.range,
        )

        mcmc_analysis.analyze(
            amisc_dir=post_train_aleatoric,
            datasets=training_datasets,
            output_dir=thruster_dir / "post_train_aleatoric",
            # plot_bands=True,
            # calc_metrics=True,
            limits=args.range,
        )

        mcmc_analysis.analyze(
            amisc_dir=prior_train,
            datasets=training_datasets,
            output_dir=thruster_dir / "prior_train",
            # plot_bands=True,
            # calc_metrics=True,
            limits=args.range,
        )

        mcmc_analysis.analyze(
            amisc_dir=prior_test,
            datasets=test_datasets,
            output_dir=thruster_dir / "prior_test",
            # plot_bands=True,
            # calc_metrics=True,
            limits=args.range,
        )

        mcmc_analysis.analyze(
            amisc_dir=post_test,
            datasets=test_datasets,
            output_dir=thruster_dir / "post_test",
            # plot_bands=True,
            # calc_metrics=True,
            limits=args.range,
        )

    prior_test_local = thruster_dir / "prior_test"
    post_test_local = thruster_dir / "post_test"
    prior_train_local = thruster_dir / "prior_train"
    post_train_local = thruster_dir / "post_train_epistemic"

    if "predict" in plot_types:
        predict_dir = thruster_dir / "predict"
        _make_prediction_plots(prior_test_local, post_test_local, predict_dir, logger, "test", thruster)
        _make_prediction_plots(prior_train_local, post_train_local, predict_dir, logger, "train", thruster)

    if "metrics" in plot_types:
        _make_metric_tables(prior_test_local, post_test_local, thruster_dir, logger, "test", thruster)
        _make_metric_tables(prior_train_local, post_train_local, thruster_dir, logger, "train", thruster)

    consolidate_plots(
        prior_test_local,
        post_test_local,
        prior_train_local,
        post_train_local,
        thruster,
        Path("results_consolidated"),
        output_dir,
        args.range,
    )


def consolidate_plots(
    prior_test: Path,
    post_test: Path,
    prior_train: Path,
    post_train: Path,
    thruster: str,
    new_output_dir: Path,
    output_dir: Path,
    range: str = "posterior",
):
    os.makedirs(new_output_dir, exist_ok=True)
    shutil.copy(output_dir / "anom_diagram.pdf", new_output_dir)

    thruster_dir = output_dir / thruster
    out_dir = new_output_dir / thruster
    os.makedirs(out_dir, exist_ok=True)

    cases = ["train", "test"]
    metrics = ["cathode_coupling_voltage_V", "discharge_current_A", "thrust_N"]
    components = ["cathode", "thruster", "plume"]

    # Root directory
    for case in cases:
        shutil.copy(thruster_dir / f"metrics_{case}.txt", out_dir)
    shutil.copy(thruster_dir / "mdot.pdf", out_dir)

    # Prediction plots - pdf only
    predict_out = out_dir / "predict"
    os.makedirs(predict_out, exist_ok=True)

    for case in cases:
        for metric in metrics:
            target = thruster_dir / "predict" / f"{metric}_{case}.pdf"
            if target.exists():
                shutil.copy(target, predict_out)

    # Plots per case
    dirs = dict(prior_test=prior_test, post_test=post_test, prior_train=prior_train, post_train=post_train)
    for k, v in dirs.items():
        case_out = out_dir / k
        os.makedirs(case_out, exist_ok=True)

        anom_file = v / "anom.pdf"
        if anom_file.is_file():
            shutil.copy(v / "anom.pdf", case_out)

        var_table = v / "variable_table.tex"
        if var_table.is_file():
            shutil.copy(v / "variable_table.tex", case_out)

        # Corner plots
        for component in components:
            corner_plot = v / f"corner_{component}.pdf"
            if corner_plot.is_file():
                shutil.copy(corner_plot, case_out)

        # Plots vs background pressure
        for metric in metrics:
            plot = v / f"{metric}_pressure_bands.pdf"
            if plot.is_file():
                shutil.copy(plot, case_out / f"{metric}_{range}.pdf")

        # Ion velocity
        ion_vel_dir = v / "ion_velocity"
        if ion_vel_dir.is_dir():
            ion_vel_pressure = "35.00uTorr" if thruster == "spt-100" else "17.50uTorr"
            shutil.copy(ion_vel_dir / "ion_velocity_allpressures.pdf", case_out)
            shutil.copy(ion_vel_dir / f"ion_velocity_p={ion_vel_pressure}.pdf", case_out)

        # Ion current density
        jion_dir = v / "ion_current_density"
        if jion_dir.is_dir():
            jion_radius = "1.32m" if thruster == "h9" else "1.00m"
            jion_pressure = "26.10uTorr" if thruster == "h9" else "15.80uTorr"

            allradii = jion_dir / f"ion_current_density_p={jion_pressure}.pdf"
            if allradii.is_file():
                shutil.copy(allradii, case_out)

            allpressures = jion_dir / f"ion_current_density_r={jion_radius}.pdf"
            if allpressures.is_file():
                shutil.copy(allpressures, case_out)

            single_pressure = jion_dir / f"p={jion_pressure}" / f"ion_current_density_r={jion_radius}.pdf"
            if single_pressure.is_file():
                shutil.copy(single_pressure, case_out / f"ion_current_density_p={jion_pressure}_r={jion_radius}.pdf")


RELATIVE_EXP_ERRORS = dict(
    cathode_coupling_voltage_V=0.01,
    thrust_N=0.01,
    discharge_current_A=0.1,
    ion_velocity=0.05,
    ion_current_sweeps=0.2,
)

MODEL_ERRORS_PEMV0 = {
    "spt-100": dict(
        train=dict(
            cathode_coupling_voltage_V=dict(median=0.02),
            thrust_N=dict(median=0.29),
            discharge_current_A=dict(median=0.63),
            ion_velocity=dict(median=0.17),
            ion_current_sweeps=dict(median=0.49),
        ),
        test=dict(
            thrust_N=dict(median=0.30),
            discharge_current_A=dict(median=0.53),
        ),
    ),
}

SURROGATE_ERRORS_PEMV0 = {
    "spt-100": dict(
        train=dict(
            cathode_coupling_voltage_V=dict(),
            thrust_N=dict(median=0.025, mean=0.026, std=0.002),
            discharge_current_A=dict(median=0.45, mean=0.45, std=0.003),
            ion_velocity=dict(median=0.21, mean=0.21, std=0.002),
            ion_current_sweeps=dict(median=0.33, mean=0.33, std=0.003),
        ),
        test=dict(
            thrust_N=dict(median=0.07, mean=0.07, std=0.001),
            discharge_current_A=dict(median=0.4, mean=0.4, std=0.001),
        ),
    ),
}


metrics = dict(
    cathode_coupling_voltage_V=dict(tex="$V_{cc}$ [V]"),
    thrust_N=dict(tex="$T_c$ [mN]"),
    discharge_current_A=dict(tex="$I_D$ [A]"),
    ion_velocity=dict(tex="$u_{ion}$ [m/s]"),
    ion_current_sweeps=dict(tex="$j_{ion}$ [A/m$^2$]"),
)


def _make_metric_tables(prior_dir: Path, post_dir: Path, output_dir: Path, logger: Logger, case: str, thruster: str):
    # Load metric files
    with open(prior_dir / "metrics.json", "r") as fd:
        prior_metrics = json.load(fd)

    with open(post_dir / "metrics.json", "r") as fd:
        post_metrics = json.load(fd)

    rows = []

    headers = [
        "\\textbf{QoI}",
        "$\\mathbf{\\xi}$ [\\%] ",
        "\\textbf{Distribution}",
        "$\\mathbf{\\mu_{50}}$",
        "$\\mathbf{\\mu}$",
        "$\\mathbf{\\sigma}$",
        "$\\mathbf{\\mu_{50} / \\xi}$",
    ]

    header = " & ".join(headers) + " \\\\\n\\midrule"

    def fmt(x, ndigits=1, scale=100, color=False):
        if x is None:
            formatted = "-"
        elif isinstance(x, str):
            formatted = x
        else:
            x = x * scale
            r = round(x, ndigits)
            int_part = round(x)
            formatted = f"{x:.{ndigits}f}"
            if x == int_part or r == int_part:
                formatted = f"{int(r)}"

        if color:
            formatted = "\\cellcolor{lightgray}" + formatted

        return formatted

    def metric_row(prev, name, metrics, error=0.025, color=False, showerror=True):
        mu_50 = metrics.get('median')
        mean = metrics.get('mean')
        std = metrics.get('std')

        fields = [
            prev,
            fmt(error, color=color) if showerror else "",
            "\\cellcolor{lightgray}" + name if color else name,
            fmt(mu_50, color=color),
            fmt(mean, color=color),
            fmt(std, color=color),
            fmt(mu_50 / error if mu_50 is not None else None, color=color, scale=1),
        ]
        return " & ".join(fields) + " "

    for metric, details in metrics.items():
        if metric not in prior_metrics or metric not in post_metrics:
            continue

        if thruster in MODEL_ERRORS_PEMV0 or thruster in SURROGATE_ERRORS_PEMV0:
            suffix = " (this work)"
        else:
            suffix = ""

        tex_name = details['tex']
        error = RELATIVE_EXP_ERRORS[metric]
        name_space = " " * len(tex_name)

        prior = prior_metrics[metric]
        rows.append(metric_row(tex_name, "Prior" + suffix, prior, error=error))

        post = post_metrics[metric]
        rows.append(
            metric_row(name_space, f"\\textbf{{Posterior{suffix}}}", post, error=error, color=True, showerror=False)
        )

        if thruster in MODEL_ERRORS_PEMV0:
            prev = MODEL_ERRORS_PEMV0[thruster][case][metric]
            rows.append(metric_row(name_space, "Posterior (prev. work, model)", prev, error=error, showerror=False))

        if thruster in SURROGATE_ERRORS_PEMV0:
            prev = SURROGATE_ERRORS_PEMV0[thruster][case][metric]
            if prev:
                rows.append(
                    metric_row(name_space, "Posterior (prev. work, surrogate)", prev, error=error, showerror=False)
                )

        rows.append("")

    # remove last blank row
    rows = rows[:-1]

    spacing = "\\\\\n"
    contents = spacing.join(rows)

    table = "\n".join([header, contents])
    with open(output_dir / f"metrics_{case}.txt", "w") as fd:
        print(table, file=fd)


def _make_mdot_plot(thruster: Optional[str], output_dir: Path, logger: Logger) -> list[Path]:
    if thruster is None:
        logger.error("Thruster must be specified to plot mass flow rate. Skipping.")
        return []

    if thruster == "h9":
        datasets = ["um2024"]
    elif thruster == "spt-100":
        datasets = ["diamant2014", "macdonald2019"]
    else:
        assert False

    unsorted_data = hallmd.data.load(hallmd.data.get_thruster(thruster).datasets_from_names(datasets))
    data = dict(sorted(unsorted_data.items(), key=lambda t: t[0].background_pressure_torr))

    p = np.array([d.background_pressure_torr for d in data])
    mdot = np.array([d.anode_mass_flow_rate_kg_s for d in data])
    hasvel = np.array([d.ion_velocity is not None for d in data.values()])

    fig, ax = plt.subplots(dpi=200, figsize=(5.5, 4.5))
    ax.set_xscale('log')
    ax.plot(p[hasvel], mdot[hasvel], '-o', label="Velocity data", color=COLORS["red"])[0]
    ax.plot(p[~hasvel], mdot[~hasvel], '-o', label="Plume data", color=COLORS["darkblue"])[0]
    ax.plot(p, mdot, '--', color="black", label="Sorted")
    ax.legend(loc='lower left')
    ax.set_xlabel("Background pressure [Torr]")
    ax.set_ylabel("Anode mass flow rate [kg/s]")
    return mcmc_analysis.save_figure(fig, output_dir, "mdot")


def _make_anom_diagram(output_dir: Path, logger: Logger) -> list[Path]:
    fig, ax = plt.subplots(dpi=200, figsize=(5, 4.5))
    ax.minorticks_off()

    N = 200
    xmax = 3
    coords = np.linspace(0, xmax, N)
    L = 0.4
    alpha = 1.0
    beta = 0.8
    z = 1.5

    pressures = [1.0, 0.0]
    linestyles = ['--', '-']
    dz = 0.7

    ax.set_xlim(0, xmax)
    ax.set_xticks([0, z - dz, z])
    ax.set_xticklabels(["0", "", "${z}_{anom}$"])
    ax.set_xlabel("Axial coordinate [channel lengths]")

    ax.set_yticks([0, (1 - beta) * alpha, alpha])
    ax.set_yticklabels(["0", "", ""])
    ax.set_ylim(0, 1.3 * alpha)
    ax.set_ylabel("Inverse anom. Hall parameter")

    handles = []

    for pressure, style in zip(pressures, linestyles):
        z_shift = mcmc_analysis.simple_pressure_shift(pressure, dz)
        nu_anom = mcmc_analysis.scaled_gaussian_bohm(coords, alpha, beta, z, L, z_shift=z_shift)
        h = ax.plot(coords, nu_anom, linestyle=style, color='black', linewidth=2)
        handles.append(h[0])

    arrowprops = dict(arrowstyle='<|-|>', facecolor='black')

    # Draw scale for transport barrier
    x_arrow = 2.9
    hi = alpha
    lo = (1 - beta) * alpha
    ax.annotate("", xytext=(x_arrow, lo), xy=(x_arrow, hi), arrowprops=arrowprops)
    ax.text(0.99 * x_arrow, 0.5 * (lo + hi), "$\\beta_{anom}\\alpha_{anom}$", ha='right', va='center', rotation=90)

    # Draw scale for L_anom
    y_L = mcmc_analysis.scaled_gaussian_bohm(z + L, alpha, beta, z, L, z_shift=0)
    mid = z + L / 2
    ax.annotate("", xytext=(z, y_L), xy=(z + L, y_L), arrowprops=arrowprops)
    ax.text(mid, 1.05 * y_L, "$L_{anom}$", ha='center', va='bottom')

    # Draw scale for dz_anom
    y_dz = 0.8 * lo
    ax.annotate("", xytext=(z - dz, y_dz), xy=(z, y_dz), arrowprops=arrowprops)
    ax.text(z - 0.5 * dz, 0.9 * y_dz, "$\\Delta z_{anom}$", ha='center', va='top')

    # Label alpha_anom
    ax.text(3, alpha * 1.04, "$\\alpha_{anom}$", horizontalalignment='right')

    ax.legend(
        handles,
        ["High pressure", "Zero pressure"],
        loc='upper left',
        ncols=2,
        columnspacing=0.5,
        handlelength=1.0,
        mode="expand",
    )

    return mcmc_analysis.save_figure(fig, output_dir, "anom_diagram")


BASE_ZORDER = 5


def _plot_multiple_datasets(
    ax: Axes,
    data: list[GlobalQuantityData],
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
        inds = np.isfinite(_data.x_mean) & np.isfinite(_data.x_err)
        if not np.any(inds):
            continue

        x_mean = _data.x_mean[inds]
        x_err = _data.x_err[inds]

        y_median = _data.y_median[inds]
        y_err = (_data.y_err_lo[inds], _data.y_err_hi[inds])

        h_err = ax.errorbar(
            x_mean,
            y_median,
            xerr=x_err,
            yerr=y_err,
            color=COLORS[_color],
            fmt='none',
            alpha=alpha,
            zorder=BASE_ZORDER + 2 * i,
        )

        h_marker = ax.scatter(
            x_mean,
            y_median,
            s=(_markersize) ** 2,
            color=COLORS[_color],
            alpha=1.0,
            linewidth=0,
            zorder=BASE_ZORDER + 2 * i + 1,
            marker=_marker,
        )
        handles.append((h_err, h_marker))

        xmin = min(xmin, np.min(x_mean - x_err))
        xmax = max(xmax, np.max(x_mean + x_err))
        ymin = min(ymin, np.min(y_median - y_err[0]))
        ymax = max(ymax, np.max(y_median + y_err[1]))

    return handles, (xmin, xmax), (ymin, ymax)


def pad_lims(lims: tuple[float, float], pad: float) -> tuple[float, float]:
    lo, hi = lims
    diff = pad * (hi - lo)
    return lo - diff, hi + diff


def plot_prediction(
    prior: GlobalQuantityData,
    posterior: GlobalQuantityData,
    quantity: str,
    output_path: Path,
    tag: str,
    marker_scale: float,
    alpha: float,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

    data = [prior, posterior]
    labels = ['Prior', 'Posterior']
    colors = ['lightorange', 'blue']
    markers = ['o', '^']
    markersizes = [4.0, 5.0]
    markersizes = [size * marker_scale for size in markersizes]

    handles, xlims, _ = _plot_multiple_datasets(ax, data, colors, markers, markersizes, alpha)

    if not handles:
        plt.close(fig)
        return []

    xlims = pad_lims(xlims, 0.1)
    vals = list(xlims)

    ax.plot(vals, vals, color='black', zorder=BASE_ZORDER + 2 * len(data) + 1, linestyle='--', linewidth=2)

    qty_name = QUANTITY_NAMES[quantity]
    ax.set_xlabel("Experimental " + qty_name)
    ax.set_ylabel("Predicted " + qty_name)
    ax.set_xlim(xlims)

    ax.legend(handles, labels, loc='upper left', markerscale=2.0)

    tag_str = f"_{tag}" if tag else ""
    return mcmc_analysis.save_figure(fig, output_path, quantity + tag_str)


def try_plot_prediction(
    prior: PathLike,
    posterior: PathLike,
    quantity: str,
    output_path: Path,
    tag: str,
    marker_scale: float,
    alpha: float,
) -> list[Path]:
    data_prior = mcmc_analysis.load_global_quantity(prior, quantity)
    data_posterior = mcmc_analysis.load_global_quantity(posterior, quantity)
    if data_prior is None or data_posterior is None:
        return []

    return plot_prediction(data_prior, data_posterior, quantity, output_path, tag, marker_scale, alpha)


def _make_prediction_plots(
    prior_dir: Path | None, post_dir: Path | None, output_dir: Path, logger: Logger, tag: str = "", thruster: str = ""
) -> list[Path]:
    plots = []
    if prior_dir is None:
        logger.error("No prior directory specified for prediction plots. Skipping.")
        return plots

    if post_dir is None:
        logger.error("No posterior directory specified for prediction plots. Skipping.")
        return plots

    if thruster == "h9":
        marker_scale = 2.0
        alpha = 0.8
    else:
        marker_scale = 1.0
        alpha = 0.5

    quantities = ["thrust_N", "discharge_current_A", "cathode_coupling_voltage_V"]
    for quantity in quantities:
        plot_paths = try_plot_prediction(
            prior_dir, post_dir, quantity, output_dir, tag, marker_scale=marker_scale, alpha=alpha
        )
        plots.append(plot_paths)

    return plots


if __name__ == "__main__":
    args = parser.parse_args()

    logger.setLevel(args.loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(args.loglevel)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:\t%(message)s"))
    logger.addHandler(ch)

    main(args)
