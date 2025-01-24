import math
import os
import pickle
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from amisc import System

import hallmd.data
from hallmd.data import OperatingCondition, ThrusterData, spt100

Dataset: TypeAlias = dict[OperatingCondition, ThrusterData]

plt.rcParams.update({"font.size": '14'})


def analyze_mcmc(path, config):
    mcmc_path = Path(path) / "mcmc"
    logfile = mcmc_path / "mcmc.csv"

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

            if not accept:
                continue

            ids.append(fields[id_ind])
            logposts.append(float(fields[log_post_ind]))
            samples.append(np.array([float(x) for x in fields[var_start:var_end]]))

    num_accept = len(samples)
    p_accept = num_accept / total_samples

    samples = np.array(samples)
    logposts = np.array(logposts)
    map_ind = np.argmax(logposts)

    map = samples[map_ind]
    empirical_cov = np.cov(samples.T)

    print(f"{p_accept=}")
    # print(f"{map=}")
    # print(f"empirical_cov = {repr(empirical_cov)}")

    ids_int = [int(i) for i in ids]

    plot_path = mcmc_path / "plots"
    os.makedirs(plot_path, exist_ok=True)

    plot_traces(system, ids_int, variables, samples, total_samples, plot_path)

    data = hallmd.data.load(spt100.macdonald2019() + spt100.diamant2014())
    results = load_sim_results(ids, mcmc_path)

    plot_result(data, results[0], "init", plot_path)
    plot_result(data, results[map_ind], "best", plot_path)

    return data, results


def load_sim_results(ids, mcmc_path: Path):
    data = []
    for id in ids:
        amisc_path = mcmc_path / id / "pemv1.pkl"
        with open(amisc_path, "rb") as f:
            data.append(pickle.load(f))

    return data


def plot_result(data, result, id, plot_path):
    input = result['input']
    output = result['output']
    plot_u_ion(data, output, 0.025, id, plot_path)
    plot_j_ion(data, output, id, plot_path)
    plot_global_quantities(data, output, id, plot_path)
    return


def plot_global_quantities(data: Dataset, output: Dataset, plot_name: str, plot_path: Path):
    global_quantities_to_plot = ['discharge_current_A', 'thrust_N', 'cathode_coupling_voltage_V']
    full_names = ["Discharge current [A]", "Thrust [mN]", "Cathode coupling voltage [V]"]
    scales = [1, 1000, 1]
    limits = [(3, 7), (70, 110), (25, 35)]

    for quantity, name, scale, limit in zip(global_quantities_to_plot, full_names, scales, limits):
        sim_qty = np.array(
            [
                getattr(x, quantity).mean * scale
                for x, dat in zip(output.values(), data.values())
                if getattr(dat, quantity) is not None
            ]
        )
        data_qty = np.array([_qty for x in data.values() if (_qty := getattr(x, quantity)) is not None])

        if len(data_qty) == 0 or len(sim_qty) == 0:
            print(f"{quantity=} has length 0")
            continue

        data_qty_mean = np.array([x.mean for x in data_qty]) * scale
        data_qty_std = np.array([x.std for x in data_qty]) * scale

        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.set_xlabel('Data')
        ax.set_ylabel('Model')
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_title(name)

        ax.plot(limits, limits, color='grey', linestyle='--')
        ax.scatter(data_qty_mean, sim_qty, color='black')
        ax.errorbar(data_qty_mean, sim_qty, xerr=2 * data_qty_std, color='black', capsize=2, linestyle='')

        fig.tight_layout()
        fig.savefig(plot_path / f"{quantity}_{plot_name}.png")

    plt.close()
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

        if output is not None:
            result = output[opcond]
            theta_rad_sim, j_sim = result.ion_current_density_coords_rad, result.ion_current_density_A_m2

            if theta_rad_sim is not None and j_sim is not None:
                ax.plot(theta_rad_sim * rad2deg, j_sim.mean, color=colors(color_scale))

        color_ind += 1

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, borderaxespad=0)
    ax_legend.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path / f"j_ion_{plot_name}.png", dpi=300)
    plt.close()

    plt.close()


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
    plt.close()
    return fig, ax


def plot_traces(system, ids, names, samples, max_samples, dir: Path = Path(".")):
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

    inputs = system.inputs()
    tex_names = [inputs[name].tex for name in names]
    perm = [i for i, _ in sorted(enumerate(tex_names), key=lambda x: x[1])]

    for col in range(num_cols):
        for row in range(num_rows):
            ax = axes[row, col]
            index = perm[row + num_rows * col]

            if row == num_rows - 1:
                ax.set_xlabel("Iteration")
            ax.set_ylabel(tex_names[index])
            ax.set_xlim(0, max_samples)

            iters = np.arange(max_samples)
            ys = np.zeros(len(iters))

            for i, id in enumerate(ids):
                if i + 1 < len(ids):
                    next_id = ids[i + 1]
                    ys[id:next_id] = samples[i, index]
                else:
                    ys[id:] = samples[i, index]

            ax.plot(iters, ys, color='black')

    plt.tight_layout()
    outpath = dir / "traces.png"
    fig.savefig(outpath)


if __name__ == "__main__":
    dir = "scripts/pem_v1/amisc_1000"
    config = "pem_v1_SPT-100.yml"
    data, results = analyze_mcmc(dir, config)
