"""`run_mcmc.py`
This script is used to run MCMC on the cathode-thruster-plume system.

Requires a valid YAML config file specifying the system.
The outputs are written to a folder called pem_{TIMESTAMP}/mcmc if no output directory is specified.
That folder in turn contains folders (numbered 000000 - num_samples) that hold the model outputs for each sample generated during MCMC.
Additionally, each sample, its log-pdf, and whether it was accepted are written to a file called samples.csv in the mcmc directory.

For usage details and a full list of options , run 'pdm run scripts/run_mcmc.py --help'
"""  # noqa: E501
import argparse
from argparse import Namespace
import copy
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Type, cast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

import amisc.distribution as distributions

from pem_core import PEM, ArrayLike, PathLike
from pem_core.data import DataEntry, extract_data_arrays, interpolate_data_instance, load_multiple_datasets
from pem_core.sampling import relative_gaussian_likelihood, LikelihoodType, DRAMSampler, PriorSampler, PreviousRunSampler

# Import data loading configuration from the Hall thruster PEM
from hallmd.data import HT_OP_VARS, HT_RENAME_MAP, HT_COORDS, HT_QOIS, OPCOND_SHORT_NAMES, pem_to_xarray

parser = argparse.ArgumentParser(description="Run MCMC calibration for the Hall thruster PEM.")

parser.add_argument(
    "config",
    type=str,
    help="the path to the `amisc` YAML config file with model and input/output variable information.",
)

parser.add_argument(
    "--executor",
    type=str,
    choices=["process", "thread"],
    default="process",
    help="the parallel executor for running MCMC samples.",
)

parser.add_argument(
    "--max-workers",
    type=int,
    default=None,
    help="the maximum number of workers to use for parallel processing. Defaults to using maxnumber of available CPUs.",
)

parser.add_argument(
    "--max-samples",
    type=int,
    default=1,
    help="The maximum number of samples to generate using MCMC",
)

parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=None,
    help="A list of file paths pointing to datasets",
)

parser.add_argument(
    "--output-interval",
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

parser.add_argument("--output-dir", type=str, default=None, help="Directory into which output files are written")

parser.add_argument(
    "--init-sample",
    type=str,
    help="CSV file containing initial sample. Only used with the `dram` sampler.",
)

parser.add_argument(
    "--init-cov",
    type=str,
    help="CSV file containing initial covariance matrix in lower triangular form. Only used with the `dram` sampler.",
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
    This is only used when using the `prev-run` sampler.
    """,
)

parser.add_argument(
    "--burn-fraction",
    type=float,
    default = 0.5,
    help="What percentage of samples from the previous run should be discarded as burn-in."
)

parser.add_argument(
    "--sample-aleatoric",
    action='store_true',
    help="Sample aleatoric varibles from distributions in config file instead of using fixed values.",
)

#%%
@dataclass
class ExecutionOptions:
    """
    Container for various options needed to execute the PEM during MCMC, including model fidelity, parallel execution settings, and some random params.
    TODO: this could probably be bundled up a bit nicer.
    """
    executor: Type
    max_workers: int
    fidelity: str | tuple | dict = (0, 0)
    noise_std: float = 0.05
    sample_aleatoric: bool = False
    print_likelihood: bool = True

def load_pem_and_opts(args: Namespace) -> tuple[PEM, ExecutionOptions]:
    """Load a PEM from an amisc config file and set up the execution options based on the command line args"""
    config = args.config
    pem = PEM.from_file(config, output_dir=args.output_dir)
    pem.set_logger(stdout=True)
    assert pem.root_dir is not None, "PEM root directory must be set. This should have been done in PEM.from_file"

    # Copy config file into output dir
    if Path(config).name not in os.listdir(pem.root_dir):
        shutil.copy(config, pem.root_dir)

    match args.executor.lower():
        case "thread":
            executor = ThreadPoolExecutor
        case "process":
            executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    opts = ExecutionOptions(
        executor,
        max_workers=args.max_workers,
        fidelity=(0, args.ncharge - 1),
        sample_aleatoric=args.sample_aleatoric,
        noise_std=args.noise_std,
    )

    return pem, opts

def _sample_aleatoric(sample_dict, system):
    """Sample the aleatoric variables using normal distributions, rather than the uniform distributions that amisc uses for Relative."""
    aleatoric_vars = ["P_b", "V_a", "mdot_a"]
    for var in aleatoric_vars:
        nominal = sample_dict[var]
        input = system.inputs()[var]
        dist = input.distribution
        assert isinstance(dist, distributions.Relative)

        # we assume the error specified in the config file is 1 standard deviation
        percent = dist.dist_args[0] / 100

        for i in range(np.atleast_1d(nominal).size):
            nom = input.denormalize(nominal[i])
            sample = nom * (1 + percent * np.random.randn())
            nominal[i] = input.normalize(sample)

        sample_dict[var] = nominal
    return sample_dict


def _consolidate_outputs(path: Path):
    """
    By default, amisc creates a directory with Thruster, Cathode, and Plume folders. The latter two are empty, but the
    former has individual files for each individual Hall thruster simulation. This creates a ton of individual files
    that can quickly deplete the maximum allowance of the Great Lakes scratch partition. This script deletes the empty
    folders, consolidates all output files into a single file, and deletes the individual files. Additionally, for
    failed runs the containing folder may be empty, so we delete it.
    """
    # Check if folder is empty
    if not os.listdir(path):
        shutil.rmtree(path)

    cathode_path = path / "Cathode"
    plume_path = path / "Plume"
    thruster_path = path / "Thruster"

    if thruster_path.is_dir():
        jsons = [f for f in thruster_path.iterdir() if f.suffix == ".json"]

        # Read JSON files and consolidate them into a single file
        json_contents = []
        for json_file in jsons:
            with open(json_file, "r") as fd:
                contents = json.load(fd)
                json_contents.append(contents)

        with open(path / "thruster.json", "w") as fd:
            json.dump(json_contents, fd)

    # Delete unneeded files
    if cathode_path.is_dir():
        cathode_path.rmdir()

    if plume_path.is_dir():
        plume_path.rmdir()

    if thruster_path.is_dir():
        shutil.rmtree(thruster_path)

def _run_model(
    params: dict[str, ArrayLike],
    pem: PEM,
    operating_conditions: list[dict[str, float]],
    base_params: dict[str, ArrayLike],
    output_dir: PathLike | None,
    opts: ExecutionOptions,
) -> list[DataEntry] | None:
    """
    Run the Hall thruster PEM at a list of operating conditions for a set of calibration params (params),
    leaving the remaining parameters equal to the values in `base_params`.
    """
    # Start with nominal values for all parameters, and then update with the current sample values.
    sample_dict = copy.deepcopy(base_params)
    sample_dict.update(params)

    # Add aleatoric variables if requested
    if opts.sample_aleatoric:
        sample_dict = _sample_aleatoric(sample_dict, pem)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Run model
    with opts.executor(max_workers=opts.max_workers) as exec:
        amisc_outputs = pem.predict(
            sample_dict,
            use_model=opts.fidelity,
            model_dir=output_dir,
            executor=exec,
            verbose=False,
        )

    # Check for errors
    if "errors" in amisc_outputs:
        raise ChildProcessError(amisc_outputs["errors"][0]['error'])

    # Get plume sweep radii
    if "Plume" in [c.name for c in pem.components]:
        sweep_radii = pem['Plume'].model_kwargs['sweep_radius']
    else:
        sweep_radii = np.array([1.0])

    pem_output = pem_to_xarray(operating_conditions, cast(dict, amisc_outputs), sweep_radii, use_corrected_thrust=True)

    if output_dir is not None:
        # Write PEM outputs to file
        if pem_output is not None :
            with open(Path(output_dir) / "pem.pkl", "wb") as fd:
                pickle.dump({"input": sample_dict, "output": pem_output}, fd)

        # Consolidate output files
        _consolidate_outputs(Path(output_dir))

    return pem_output

def _log_likelihood(
    data: list[DataEntry],
    base_params: dict[str, ArrayLike],
    opts: ExecutionOptions,
) -> LikelihoodType:
    """
    Return a log-likelihood function comparing simulation outputs to data for the Hall thruster PEM
    This function gets evaluated for each MCMC sample.
    The closure approach allows us to use a consistent function signature in the Sampler interface.
    The likelihood function has signature: likeliood(pem, params, output_dir) -> float
    """
    def _likelihood_func(pem: PEM, params: dict[str, ArrayLike], output_dir: PathLike | None) -> float:
        opconds = [d.operating_condition for d in data]
        sim_results = _run_model(params, pem, opconds, base_params, output_dir, opts)

        if sim_results is None:
            return -np.inf

        # Interpolate sim results to data coordinates
        sim_itp = [interpolate_data_instance(_sim.data, _data.data) for (_sim, _data) in zip(sim_results, data)]

        # Extract data per-field into 1-D vectors
        sim_arrays = extract_data_arrays(sim_itp)
        data_arrays = extract_data_arrays(data)

        # Extract likelihood and L-2 distances for each component of the QoI (e.g. thrust, ion velocity, etc.)
        component_stats = {}
        L = 0.0
        for field_name, (data_vec, _) in data_arrays.items():
            sim_vec, _ = sim_arrays[field_name]
            distance, likelihood = relative_gaussian_likelihood(data_vec, sim_vec, opts.noise_std)        
            component_stats[field_name] = (distance, likelihood)
            L += likelihood

        # Print table to output with likelihoods and distances to stdout
        if opts.print_likelihood:
            name_len = max(len(n) for n in component_stats.keys()) + 1
            indent = "  "
            header = indent + " Field" + (" " * (name_len - 5)) + "|  L2 norm   | Likelihood  "
            rule = "-" * len(header)
            print("\n" + rule)
            print(header)
            print(rule)
            for name, (l2, logp) in component_stats.items():
                pad = name_len - len(name)
                print(indent + f" {name}{' ' * pad}| {l2:10.5f} | {logp:10.5f}")
            print(rule + "\n")
        return L

    return _likelihood_func

def main(args):
    # Load PEM from YAML file and determine nominal and calibration parameters
    pem, exec_opts = load_pem_and_opts(args)
    component_names = set([c.name for c in pem.components])
    base_vars = pem.get_nominal_inputs(norm=True)
    calibration_vars = pem.get_inputs_by_category("calibration", sort="name")

    # Load data from files
    data = load_multiple_datasets(args.datasets, HT_OP_VARS, HT_QOIS, HT_COORDS, HT_RENAME_MAP)
    operating_conditions = [d.operating_condition for d in data]

    print(f"Calibration variables:\n\t{", ".join([p.name for p in calibration_vars])}")
    print(f"Number of operating conditions: {len(operating_conditions)}")

    # Load operating conditions into input dictionary
    for (long_name, param) in OPCOND_SHORT_NAMES.items():
        if param not in pem.inputs():
            base_vars[param] = 1.0
        else:
            inputs_unnorm = np.array([cond[long_name] for cond in operating_conditions])
            base_vars[param] = pem.inputs()[param].normalize(inputs_unnorm) # type: ignore

    # Remove data where theta < 0 for ion current density
    JION_KEY = "ion current density"
    for d in data:
        if (jion := d.data.get(JION_KEY)) is not None:
            jion.val = jion.val.sel(theta=slice(0, np.pi/2))
            if (jion.err is not None):
                jion.err = jion.err.sel(theta=slice(0, np.pi/2))

    # Set plume sweep radii based on data.
    # If there is no ion current density in the data, we just set the sweep radius to 1.0 (i.e. no sweeping).
    # This allows us to still run MCMC on cases where there is no ion current density data without having to change the model config.
    if "Plume" in component_names:
        sweep_radii = []
        for entry in data:
            if JION_KEY in entry.data and (ji_coords := entry.data[JION_KEY].val.coords) is not None and "r" in ji_coords:
                sweep_radii.extend(ji_coords["r"].values)

        pem['Plume'].model_kwargs['sweep_radius'] = np.sort(np.unique(sweep_radii)) if sweep_radii else np.array([1.0])

    # Set up output dir
    output_dir = Path(pem.root_dir) / "mcmc" # type: ignore
    os.makedirs(output_dir, exist_ok=True)

    # Set up sampler
    log_likelihood = _log_likelihood(data, base_vars, exec_opts)
    sampler_args = dict(pem=pem, sample_vars=calibration_vars, base_vars=base_vars, log_likelihood=log_likelihood, output_dir=output_dir)

    match args.sampler:
        case "dram":
            sampler = DRAMSampler(**sampler_args, init_sample_file=args.init_sample, init_cov_file=args.init_cov)
        case "prior":
            sampler = PriorSampler(**sampler_args)
        case "prev-run":
            sampler = PreviousRunSampler(args.prev, burn_fraction=args.burn_fraction, **sampler_args)
        case _:
            raise ValueError(f"Invalid sampler {args.sampler} specified. This should be unreachable.")

    # MCMC loop
    sampler.sample(args.max_samples)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)