from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.floating[Any]]
type PathLike = str | Path


@dataclass(frozen=True)
class OperatingCondition:
    P_B: float
    V_a: float
    mdot_a: float


@dataclass(frozen=True)
class Measurement[T]:
    mean: T
    std: T

    def __str__(self):
        return f"(μ = {self.mean}, σ = {self.std})"


@dataclass
class ThrusterData:
    # Cathode
    V_cc: Measurement[float] | None = None
    # Thruster
    T: Measurement[float] | None = None
    I_D: Measurement[float] | None = None
    I_B0: Measurement[float] | None = None
    eta_c: Measurement[float] | None = None
    eta_m: Measurement[float] | None = None
    eta_v: Measurement[float] | None = None
    eta_a: Measurement[float] | None = None
    uion_coords: Array | None = None
    uion: Measurement[Array] | None = None
    # Plume
    div_angle: Measurement[float] | None = None
    radius: float | None = None
    jion_coords: Array | None = None
    jion: Measurement[Array] | None = None

    def __str__(self) -> str:
        indent = "\t"
        out: str = "ThrusterData(\n"
        for field in fields(ThrusterData):
            val = getattr(self, field.name)
            if val is not None:
                out += f"{indent}{field.name} = {val},\n"

        out += ")\n"
        return out


def load(files: Sequence[PathLike] | PathLike) -> dict[OperatingCondition, ThrusterData]:
    data: dict[OperatingCondition, ThrusterData] = {}
    if isinstance(files, Sequence):
        # Recursively load resources in this list (possibly list of lists)
        for file in files:
            data.update(load(file))
    else:
        data.update(_load_dataset(files))

    return data


def _load_dataset(file: PathLike) -> dict[OperatingCondition, ThrusterData]:
    table = _table_from_file(file, delimiter=",", comments="#")
    data: dict[OperatingCondition, ThrusterData] = {}

    # Compute anode mass flow rate, if not present
    mdot_a_key = "anode flow rate (mg/s)"
    mdot_t_key = "total flow rate (mg/s)"
    flow_ratio_key = "anode-cathode flow ratio"
    keys = list(table.keys())

    if mdot_a_key in keys:
        mdot_a = table[mdot_a_key]
    else:
        if mdot_t_key in keys and flow_ratio_key in keys:
            flow_ratio = table[flow_ratio_key]
            anode_flow_fraction = flow_ratio / (flow_ratio + 1)
            mdot_a = table[mdot_t_key] * anode_flow_fraction

    # Get background pressure and discharge voltage
    P_B = np.log10(table["background pressure (torr)"])
    V_a = table["anode voltage (v)"]

    num_rows = len(table[keys[0]])
    row_num = 0
    opcond_start_row = 0
    opcond = OperatingCondition(P_B[0], V_a[0], mdot_a[0])

    while True:
        if row_num < num_rows:
            next_opcond = OperatingCondition(P_B[row_num], V_a[row_num], mdot_a[row_num])
            if next_opcond == opcond:
                row_num += 1
                continue

        # either at end of operating condition or end of table
        # fill up thruster data object for this row
        data[opcond] = ThrusterData()

        # Fill up data
        # We assume all errors (expressed as value +/- error) correspond to two standard deviations
        for key, val in table.items():
            if key == "thrust (mn)":
                # Load thrust data
                T = val[opcond_start_row] * 1e-3  # convert to Newtons
                T_std = table["thrust relative uncertainty"][opcond_start_row] * T / 2
                data[opcond].T = Measurement(mean=T, std=T_std)

            elif key == "anode current (a)":
                # Load discharge current data
                # assume a 0.1-A std deviation for discharge current
                data[opcond].I_D = Measurement(mean=val[opcond_start_row], std=0.1)

            elif key == "cathode coupling voltage (v)":
                # Load cathode coupling data
                V_cc = val[opcond_start_row]
                V_cc_std = table["cathode coupling voltage absolute uncertainty (v)"][opcond_start_row] / 2
                data[opcond].V_cc = Measurement(mean=V_cc, std=V_cc_std)

            elif key == "ion velocity (m/s)":
                # Load ion velocity data
                uion: Array = val[opcond_start_row:row_num]
                uion_std: Array = table["ion velocity absolute uncertainty (m/s)"][opcond_start_row:row_num] / 2
                data[opcond].uion = Measurement(mean=uion, std=uion_std)
                data[opcond].uion_coords = table["axial position from anode (m)"][opcond_start_row:row_num]

            elif key == "ion current density (ma/cm^2)":
                # Load ion current density data
                jion: Array = val[opcond_start_row:row_num] * 10  # Convert to A / m^2
                jion_std: Array = table["ion current density relative uncertainty"][opcond_start_row:row_num] * jion / 2
                r = table["radial position from thruster exit (m)"][0]
                jion_coords: Array = table["angular position from thruster centerline (deg)"][opcond_start_row:row_num]

                # Keep only measurements at angles less than 90 degrees
                keep_inds = jion_coords < 90
                data[opcond].jion_coords = jion_coords[keep_inds] * np.pi / 180
                data[opcond].radius = r
                data[opcond].jion = Measurement(mean=jion[keep_inds], std=jion_std[keep_inds])

        # Advance to next operating condition or break out of loop if we're at the end of the table
        if row_num == num_rows:
            break

        opcond = next_opcond
        opcond_start_row = row_num

    return data


def _table_from_file(file: PathLike, delimiter=",", comments="#") -> dict[str, Array]:
    # Read header of file to get column names
    # We skip comments (lines starting with the string in the `comments` arg)
    header_start = 0
    with open(file, "r") as f:
        for i, line in enumerate(f):
            if not line.startswith(comments):
                header = line.rstrip()
                header_start = i
                break

    column_names = header.split(delimiter)
    data = np.genfromtxt(file, delimiter=delimiter, comments=comments, skip_header=header_start + 1)

    table: dict[str, Array] = {column.casefold(): data[:, i] for (i, column) in enumerate(column_names)}
    return table
