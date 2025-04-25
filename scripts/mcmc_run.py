"""`mcmc.py`

This script is used to run MCMC on the cathode-thruster-plume system.

Requires a valid YAML config file specifying the system.
The outputs are written to a folder called amisc_{TIMESTAMP}/mcmc if no output directory is specified.
That folder in turn contains folders (numbered 000000 - num_samples) that hold the model outputs for each sample generated during MCMC.
Additionally, each sample, its log-pdf, and whether it was accepted are written to a file called mcmc.csv in the main mcmc directory.

For usage details and a full list of options , run 'pdm run scripts/run_mcmc.py --help'

"""  # noqa: E501

import argparse
import os

import numpy as np
import pem_mcmc as mcmc

import hallmd.data

parser = argparse.ArgumentParser(description="MCMC scripts")

parser.add_argument(
    "config_file",
    type=str,
    help="the path to the `amisc` YAML config file with model and input/output variable information.",
)

parser.add_argument(
    "--executor",
    type=str,
    default="process",
    help="the parallel executor for running MCMC samples. Options are `thread` or `process`. Default (`process`).",
)

parser.add_argument(
    "--max_workers",
    type=int,
    default=None,
    help="the maximum number of workers to use for parallel processing. Defaults to using maxnumber of available CPUs.",
)

parser.add_argument(
    "--max_samples",
    type=int,
    default=1,
    help="The maximum number of samples to generate using MCMC",
)

parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=["diamant2014", "macdonald2019"],
    help="""
    A list of datasets to use.
    For the SPT-100, pick from [diamant2014, macdonald2019, sankovic1993]".
    For the H9, pick from [um2024, gt2024].
    """,
)

parser.add_argument(
    "--output_interval",
    type=int,
    default=100,
    help="How frequently plots are generated",
)

parser.add_argument(
    "--ncharge",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="Number of ion charge states to include in simulation. Must be between 1 and 3.",
)

parser.add_argument(
    "--noise-std",
    type=float,
    default=0.05,
    help="Standard deviation of the Gaussian noise in the likelihood calculation",
)

parser.add_argument("--output_dir", type=str, default=None, help="Directory into which output files are written")

parser.add_argument(
    "--init_sample",
    type=str,
    help="""CSV file containing initial sample. Only used with the `dram` sampler.""",
)

parser.add_argument(
    "--init_cov",
    type=str,
    help="""CSV file containing initial covariance matrix. Only used with the `dram` sampler.""",
)

parser.add_argument(
    "--sampler",
    type=str,
    choices=["dram", "prior", "prev-run"],
    default="dram",
    help="""
    The type of sampler to use
    - `dram`: the delayed-rejection adaptive metropolis sampler
    - `prior`: sample from the variable prior distributions only
    - `prev-run`: draw randomly (with replacement) from the samples of a previous run.

    The `prev-run` sampler requires the --prev argument to be passed.
    """,
)

parser.add_argument(
    "--prev",
    type=str,
    help="""
    The file containing samples from a previous run using this config.
    This is only useful when using the `prev-run` sampler.
    """,
)

parser.add_argument(
    "--sample-aleatoric",
    action='store_true',
    help="Sample aleatoric varibles from distributions in config file instead of using fixed values.",
)


def _update_opts(opts, root_dir, sample_index):
    id_str = f"{sample_index:06d}"
    opts.directory = root_dir / id_str
    os.mkdir(opts.directory)


def main(args):
    system, opts = mcmc.load_system_and_opts(args)
    base = mcmc.get_nominal_inputs(system)

    # Determine thruster and load data
    thruster_name = system['Thruster'].model_kwargs['thruster']
    thruster = hallmd.data.get_thruster(thruster_name)
    datasets = thruster.datasets_from_names(args.datasets)
    data = hallmd.data.load(datasets)
    operating_conditions = list(data)

    # Load operating conditions into input dict
    operating_params = ["P_b", "V_a", "mdot_a"]
    for param in operating_params:
        long_name = hallmd.data.opcond_keys[param.casefold()]
        inputs_unnorm = np.array([getattr(cond, long_name) for cond in data.keys()])
        base[param] = system.inputs()[param].normalize(inputs_unnorm)

    # Choose calibration params
    params_to_calibrate = mcmc.load_calibration_variables(system, sort='name')

    # Set plume sweep radii based on data.
    # If there are no sweep radii in the data, we predict the current density
    # at a distance of 1 meter only.
    sweep_radii = []
    for d in data.values():
        if d.ion_current_sweeps is None:
            continue
        sweep_radii += [s.radius_m for s in d.ion_current_sweeps]
    if sweep_radii:
        sweep_radii = np.sort(np.unique(sweep_radii))
        system['Plume'].model_kwargs['sweep_radius'] = np.array(sweep_radii)
    else:
        system['Plume'].model_kwargs['sweep_radius'] = np.array([1.0])

    print(f"{params_to_calibrate=}")
    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Prepare output file
    root_dir = opts.directory
    logfile = root_dir / "mcmc.csv"
    delimiter = ","
    header = delimiter.join(["id"] + [p.name for p in params_to_calibrate] + ["log_posterior"] + ["accepted"])
    with open(logfile, "w") as fd:
        print(header, file=fd)

    # Set up directory for initial sample
    _update_opts(opts, root_dir, 0)

    # Set up samplers
    match args.sampler:
        case "dram":
            sampler = mcmc.DRAMSampler(
                params_to_calibrate,
                data,
                system,
                base,
                opts,
                init_sample_file=args.init_sample,
                init_cov_file=args.init_cov,
            )
        case "prior":
            sampler = mcmc.PriorSampler(params_to_calibrate, data, system, base, opts)
        case "prev-run":
            sampler = mcmc.PreviousRunSampler(
                params_to_calibrate, data, system, base, opts, prev_run_file=args.prev, burn_fraction=0.5
            )
        case _:
            raise ValueError("Unreachable")

    # Initialize sampler parameters and evaluate posterior on init sample
    max_samples: int = args.max_samples
    init_sample = sampler.initial_sample()
    best_logp = sampler.logpdf(init_sample)
    mcmc.append_sample_row(logfile, 0, init_sample, best_logp, True)
    num_accept = 1

    if isinstance(sampler, mcmc.DRAMSampler):
        burn_fraction = 0.5
        num_subsample = 1000
    else:
        burn_fraction = 0.0
        num_subsample = None

    # Main sampler loop
    start_index = 1
    _update_opts(opts, root_dir, start_index)
    for i, (sample, logp, accepted_bool) in enumerate(sampler):
        mcmc.append_sample_row(logfile, start_index + i, sample, logp, accepted_bool)

        if logp > best_logp:
            best_logp = logp

        if accepted_bool:
            num_accept += 1

        print(
            f"sample: {i + start_index}/{max_samples}, logp: {logp:.3f}, best logp: {best_logp:.3f},",
            f"accepted: {accepted_bool}, p_accept: {num_accept / (i + 1 + start_index) * 100:.1f}%",
        )

        corner = i == max_samples or (i > 100 and (i % (args.output_interval * 10) == 0))

        if (i == max_samples) or (i % args.output_interval == 0):
            mcmc.analyze(
                root_dir.parent,
                args.datasets,
                plot_corner=corner,
                plot_bands=True,
                plot_traces=True,
                calc_metrics=True,
                save_restart=True,
                proposal_cov=sampler.cov(),
                subsample=num_subsample,
                burn_fraction=burn_fraction,
            )

        if i >= max_samples:
            break

        _update_opts(opts, root_dir, i + 1 + start_index)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
