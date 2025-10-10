import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pem_mcmc as mcmc
import pem_mcmc.analysis as mcmc_analysis

parser = argparse.ArgumentParser()
parser.add_argument("thruster", type=str, choices=["h9", "spt100"])
parser.add_argument("--dir", type=Path)
parser.add_argument("--name", type=str)
args = parser.parse_args()

if args.thruster == "h9":
    config = Path("scripts/pem_v1/pem_v1_H9.yml")
    device_name = "H9"
else:
    config = Path("scripts/pem_v1/pem_v1_SPT-100.yml")
    device_name = "SPT-100"

system = mcmc.load_system(config)
variables = mcmc.load_calibration_variables(system, sort='name')


def load_samples(dir: Path):
    subdirs = os.listdir(dir)
    assert len(subdirs) == 1
    mcmc_path = dir / subdirs[0] / "mcmc"
    mcmc_file = mcmc_path / "mcmc.csv"
    vars, samples, logposts, _, _ = mcmc.read_output_file(mcmc_file)
    with open(dir / subdirs[0] / "mcmc_analysis" / "metrics_raw.json", "r") as fd:
        var_dict = json.load(fd)

    return dict(vars=vars, x=np.array(samples), logp=np.array(logposts), qois=var_dict)


A = load_samples(args.dir / "A")
B = load_samples(args.dir / "B")

assert A["vars"] == B["vars"]

C = []
for i in range(len(A["vars"])):
    Ci = load_samples(args.dir / f"C_{i:02d}")
    assert Ci["vars"] == B["vars"]
    C.append(Ci)

inds = {}


def compute_indices(A, B, C, qoi="discharge_current_A", bootstrap=False):
    Ay = np.array(A["qois"][qoi])
    By = np.array(B["qois"][qoi])
    min_samples = min(len(Ay), len(By))

    for i, var in enumerate(A["vars"]):
        Cy = np.array(C[i]["qois"][qoi])[:min_samples]
        min_samples = min(min_samples, len(Cy))

    Ay = Ay[:min_samples]
    By = By[:min_samples]

    select_inds = np.logical_and(np.isfinite(Ay), np.isfinite(By))

    for i, var in enumerate(A["vars"]):
        Cy = np.array(C[i]["qois"][qoi])[:min_samples]
        select_inds = np.logical_and(select_inds, np.isfinite(Cy))

    select_inds = np.arange(len(Ay))[select_inds]

    N = len(select_inds)

    if bootstrap:
        select_inds = np.random.choice(select_inds, N, replace=True)

    Ay = Ay[select_inds]
    By = By[select_inds]

    for i, var in enumerate(A["vars"]):
        inds[var] = {}
        Cy = np.array(C[i]["qois"][qoi])[:min_samples][select_inds]

        # Compute T - total Sobol' index
        mean_b_c = np.mean(0.5 * (By + Cy))
        T_num = np.mean(Cy * By) - mean_b_c**2
        T_den = 0.5 * np.mean(By**2 + Cy**2) - mean_b_c**2
        T = 1 - (T_num / T_den)
        inds[var]["T"] = T if not (T > 1.15 and var == "Pstar") else 0.0  # catch strange Pstar outlier

    return inds


num_bootstrap = 100
QUANTILES = [0.05, 0.5, 0.95]

stats = {}

for qoi in A["qois"]:
    results = {}
    for i in range(num_bootstrap):
        sobol_inds = compute_indices(A, B, C, qoi=qoi, bootstrap=(num_bootstrap > 1))
        for k, v in sobol_inds.items():
            val = v["T"]
            if i == 0:
                results[k] = [val]
            else:
                results[k].append(val)

    print(f"QOI: {qoi}\n============")
    stats[qoi] = {}
    for k, v in results.items():
        median = np.array(v)
        qt = np.quantile(median, q=QUANTILES, axis=0)
        stats[qoi][k] = dict(lo=qt[0], med=qt[1], hi=qt[2])
        print(f"{k}: {qt[1]:.4f} ({qt[0]:.4f}-{qt[1]:.4f})")

    print()

qois = ["cathode_coupling_voltage_V", "discharge_current_A", "thrust_N", "ion_velocity", "ion_current_sweeps"]
qoi_symbols = [r"$V_{cc}$", r"$I_D$", r"$T_c$", r"$u_{ion}$", r"$j_{ion}$"]
tex_names = [var["tex"] for var in variables]
tex_names = []
for var in variables:
    tex = var["tex"]
    tex = tex.replace("anom", "an")
    tex = tex.replace("T_e", "T_{ec}")
    tex_names.append(tex)

x = np.arange(len(A["vars"]))
padding = 0.0
width = (1 - 1.5 * padding) / len(qois)
colors = mcmc_analysis.COLORS
qoi_colors = dict(
    cathode_coupling_voltage_V="red",
    discharge_current_A="blue",
    thrust_N="lightorange",
    ion_velocity="green",
    ion_current_sweeps="darkblue",
)

fig, ax = plt.subplots(1, 1, figsize=(11, 2.5), constrained_layout=True, dpi=200)
ax.set_ylabel("Sobol' index")
ax.grid(False)
ax.set_xticks(x + 0.5, tex_names)
ax.minorticks_off()

for i, qoi in enumerate(qois):
    if qoi not in stats:
        continue
    median = np.array([stats[qoi][var]["med"] for var in A["vars"]])
    hi = np.array([stats[qoi][var]["hi"] for var in A["vars"]])
    lo = np.array([stats[qoi][var]["lo"] for var in A["vars"]])

    err = [hi - median, median - lo]

    rects = ax.bar(
        x + padding + i * width,
        median,
        width,
        color=mcmc_analysis.COLORS[qoi_colors[qoi]],
        yerr=err,
        align='edge',
        label=qoi_symbols[i],
        error_kw=dict(capsize=3, linewidth=1),
    )

for i, x in enumerate(x):
    color = "gray" if i % 2 == 0 else "white"
    ax.axvspan(x, x + 1, color=color, zorder=0, alpha=0.2, linewidth=0)

ax.set_xlim(0, len(A["vars"]))
ax.set_ylim(bottom=0, top=1.3)
ax.legend(loc='upper right', ncols=len(qois), handlelength=0.8, handletextpad=0.5, columnspacing=0.5)
filename = f"sobol_{args.thruster}{'_' + args.name if args.name else ''}"
fig.savefig(f"{filename}.png")
fig.savefig(f"{filename}.pdf")
fig.savefig(f"{filename}.svg")
plt.close(fig)
