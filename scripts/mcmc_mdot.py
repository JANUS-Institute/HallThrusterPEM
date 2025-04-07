from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pem_mcmc.analysis import COLORS, RCPARAMS, save_figure

import hallmd.data

plt.rcParams.update(RCPARAMS)


def main():
    data = hallmd.data.load(hallmd.data.get_thruster("h9").datasets_from_names(["um2024"]))

    p = np.array([d.background_pressure_torr for d in data])
    mdot = np.array([d.anode_mass_flow_rate_kg_s for d in data])
    hasvel = np.array([d.ion_velocity is not None for d in data.values()])

    sortperm = np.array([t[0] for t in sorted(enumerate(p), key=lambda t: t[1])])

    fig, ax = plt.subplots(dpi=200, figsize=(5.5, 4.5))
    ax.set_xscale('log')
    ax.plot(p[hasvel], mdot[hasvel], '-o', label="Velocity data", color=COLORS["red"])[0]
    ax.plot(p[~hasvel], mdot[~hasvel], '-o', label="Plume data", color=COLORS["darkblue"])[0]
    ax.plot(p[sortperm], mdot[sortperm], '--', color="black", label="Sorted")
    ax.legend(loc='lower left')
    ax.set_xlabel("Background pressure [Torr]")
    ax.set_ylabel("Anode mass flow rate [kg/s]")
    save_figure(fig, Path("."), "mdot")


if __name__ == "__main__":
    main()
