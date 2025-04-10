"""
This script, when given an amisc output directory with MCMC results, will calculate parameter statistics and,
optionally, create corner plots for each component independently.
"""

import argparse
import json
import os
from pathlib import Path

import amisc.distribution as distributions
import amisc.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import pem_mcmc
import pem_mcmc.analysis
import scipy.stats
from amisc import System, Variable
from pem_mcmc.analysis import COLORS, RCPARAMS, pad_limits

plt.rcParams.update(RCPARAMS)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True, help="MCMC input directory to load.")
parser.add_argument("-b", "--burn-fraction", type=float, default=0.75, help="Fraction of samples to burn.")
parser.add_argument("-o", "--output", type=str, default=".", help="Output directory for plots and files.")
parser.add_argument("--plot-corner", action="store_true", help="Whether to make a corner plot.")

QUANTILES: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95]

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


def load_system(directory: Path) -> System:
    """
    Given a directory, find the first yaml file in that directory and load the amisc system from that file.
    """
    dir_contents = os.listdir(directory)
    for file_or_dir in dir_contents:
        # Skip directories
        if os.path.isdir(file_or_dir):
            continue

        # Skip files without .yml or .yaml extensions
        file = Path(file_or_dir)
        ext = file.suffix.casefold()
        if ext not in {".yml", ".yaml"}:
            continue

        # Load the system from the file
        return System.load_from_file(directory / file)

    # If we're here, we didn't find anything and should error.
    raise ValueError(f"Could not find a yaml file in directory {directory}.")


def split_vars_by_component(vars, component):
    # We explicitly call out "T_e" here so we don't double-list it in cathode and thruster
    return [
        component.inputs[v] for v in vars if v in component.inputs and not (component.name == "Thruster" and v == "T_e")
    ]


def _plot_hist_1D(axis, samples, lims: tuple[float, float], N: int = 100):
    kernel = scipy.stats.gaussian_kde(samples)
    xvals = np.linspace(lims[0], lims[1], N)
    probs = kernel(xvals)
    axis.fill_between(xvals, np.zeros(N), probs, color=COLORS["lightblue"])
    axis.plot(xvals, probs, linewidth=3, color=COLORS["blue"])


def plot_component_corner(component_name: str, component_samples: dict[Variable, np.ndarray], output_dir: Path):
    num_vars = len(component_samples.keys())
    subfig_size = 2.0
    rows = num_vars

    fig, axes = plt.subplots(
        rows,
        rows,
        dpi=200,
        figsize=(rows * subfig_size, rows * subfig_size),
        layout="constrained",
    )

    limits = [pad_limits((np.min(samples), np.max(samples)), 0.2) for samples in component_samples.values()]

    for irow, rowvar in enumerate(component_samples.keys()):
        for icol, colvar in enumerate(component_samples.keys()):
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
                _plot_hist_1D(axis, component_samples[colvar], limits[icol])
                axis.set_yticks([])
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.set_ylim(bottom=0)
                continue
            else:
                axis.hexbin(component_samples[colvar], component_samples[rowvar], gridsize=12, mincnt=1)
                axis.set_ylim(limits[irow])

    pem_mcmc.analysis.save_figure(fig, output_dir, "corner_" + component_name.casefold(), tight_layout=False)


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


def main(args: argparse.Namespace):
    directory = Path(args.dir)

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    file = directory / "mcmc" / "mcmc.csv"
    vars, samples, _, _, _ = pem_mcmc.read_output_file(file)
    index_map_sorted = {var: i for (i, var) in enumerate(vars)}

    num_burn = round(args.burn_fraction * len(samples))
    samples = samples[num_burn:]
    samples = np.array(samples)

    # Split samples by variables.
    vars = sort_order
    var_name_dict = {var: samples[:, index_map_sorted[var]] for var in vars}

    # Load system and components
    system = load_system(directory)

    # Compute statistics
    system_vars = {system.inputs()[var]: _samples for (var, _samples) in var_name_dict.items()}
    compute_sample_statistics(system_vars, output_dir)

    if args.plot_corner:
        # Split up samples by variables for each component
        vars = {comp.name: split_vars_by_component(vars, comp) for comp in system.components}
        component_samples = {
            comp.name: {v: var_name_dict[v.name] for v in vars[comp.name]} for comp in system.components
        }

        for component_name, samples in component_samples.items():
            plot_component_corner(component_name, samples, output_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
