from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt

from hallmd.data import OperatingCondition, ThrusterData

Dataset: TypeAlias = dict[OperatingCondition, ThrusterData]


# Dummy device for now, will interface with real device class later
@dataclass
class Device:
    L_ch: float


def plot_u_ion(output: Dataset | None, data: Dataset, device: Device, plt_name: str):
    colors = ["red", "green", "blue"]
    fig, ax = plt.subplots()
    _ = ax.set_xlim(0, 3.0)
    _ = ax.set_xlabel("$z / L_{ch}$")
    _ = ax.set_ylabel("$u_{ion, z}$ [km/s]")

    for i, (opcond, dataset) in enumerate(data.items()):
        pressure_base = round(1e5 * opcond.background_pressure_torr, 1)

        z = dataset.ion_velocity_coords_m
        u_ion = dataset.ion_velocity_m_s

        if z is None or u_ion is None:
            continue

        label = f"$P_B = {pressure_base}\\times 10^{{-5}}$ Torr"
        ax.errorbar(
            z / device.L_ch,
            u_ion.mean / 1000,
            yerr=2 * u_ion.std / 1000,
            label=label,
            color=colors[i],
            fmt="--o",
            markersize=4.0,
        )

        if output is None:
            continue

        result = output[opcond]
        z_sim, u_sim = result.ion_velocity_coords_m, result.ion_velocity_m_s

        if z_sim is not None and u_sim is not None:
            plt.plot(z_sim / device.L_ch, u_sim.mean / 1000, color=colors[i])

    plt.legend()
    plt.tight_layout()
    fig.savefig(f"u_ion_{plt_name}.png", dpi=300)
    return fig, ax
