import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from amisc import System

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

    empirical_cov = np.cov(samples.T)
    print(repr(empirical_cov))

    plot_traces(system, ids, variables, samples, total_samples, mcmc_path)


def plot_traces(system, ids, names, samples, max_samples, dir: Path = Path(".")):
    _, num_vars = samples.shape
    ids_int = [int(i) for i in ids]

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

            for i, id in enumerate(ids_int):
                if i + 1 < len(ids_int):
                    next_id = ids_int[i + 1]
                    ys[id:next_id] = samples[i, index]
                else:
                    ys[id:] = samples[i, index]

            ax.plot(iters, ys, color='black')

    plt.tight_layout()
    fig.savefig(dir / "traces.png")


if __name__ == "__main__":
    dir = "scripts/pem_v1/amisc_2025-01-23T19.03.03"
    config = "pem_v1_SPT-100.yml"
    analyze_mcmc(dir, config)
