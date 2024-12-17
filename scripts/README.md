This directory is used for running UQ analysis on different PEM configurations and experimental data.

## Predictive engineering models (PEM)

We define a "PEM" as a set of coupled component models that predicts global performance metrics and quantities of interest (QoIs), such as thrust, ion velocity, and efficiency, as a function of global inputs, such as discharge voltage, mass flow rate, and background chamber pressure. 

We currently define two PEM configurations:

- **PEM test** - a temporary configuration for rapid iteration testing of variables, models, etc.
- **PEM v0** - a purely feedforward coupling between a semi-empirical cathode model, a 1d fluid thruster model, and a semi-empirical plume model ([Eckels et al. 2024](https://rdcu.be/dVmim)).

See the corresponding directories for config files and analysis scripts.

A PEM must define the sets of parameters and ranges over which it seeks to be applicable. Any significant change to the parameters, underlying models, or the addition of new models or parameters would warrant a new "pem_vXX" directory for analysis. The underlying models themselves (rather than the specific parameters or configurations) should be maintained in the `src/hallmd` Python package. The `hallmd` package also maintains devices, experimental data, and utility functions for each.

## Scripts
There are a few general scripts that can be used for any PEM configuration:

### Environment setup
Installs `pdm` and the `hallmd` package. Also installs Julia and `HallThruster.jl`.
```shell
source setup_env.sh --hallthruster-version=0.17.2 --julia-version=1.10
```

### Train a surrogate
Loads a surrogate configuration from file, generates compression and test set data, trains the surrogate, and plots diagnostics.
```shell
./train.sh pem_v0/pem_v0_SPT-100.yml --compression_samples=100 --test_samples=100 --max_iter=100 --executor=thread
```

## PEM analysis procedure
For a new PEM configuration, follow this procedure for analysis:

1. Define a single `.yml` configuration file that specifies all models, parameters, configurations, etc. The format of this file follows the [`amisc` interface](https://eckelsjd.github.io/amisc/guides/config_file/). See also the config file for [PEM v0](pem_v0/pem_v0_SPT-100.yml) for an example.
1. Generate compression data for field quantities and a random test set for evaluating surrogate performance. See the [`gen_data.py`](gen_data.py) script.
1. Train a surrogate to approximate the model. See the [`fit_surr.py`](fit_surr.py) script.
1. Evaluate the performance of the surrogate. For example, you can [plot 1d slices](plot_slice.py) over the inputs and compare to the true model. Generally, < 10\% error is advisable before moving forward.
1. For convenience, `gen_data`, `fit_surr`, and `plot_slice` are all combined into one step using [`train.sh`](train.sh)).
1. Calibrate the model parameters to fit experimental data using the surrogate as a cheap approximation. See the [MCMC calibration](pem_v0/mcmc.py) of PEM v0.
1. Run any other desired UQ analyses (Monte Carlo, Sobol' sensitivity analysis, etc.). The [PEM v0 directory](pem_v0) again provides a few examples.

In general, anytime _anything_ changes about the models (including parameter ranges, values, configurations, or general computations), then a new surrogate _must_ be built, including new compression data if applicable and a new test set. Note that individual component models can be modified and retrained independently of other components.

Analysis scripts that are uniquely tailored to a specific PEM configuration (i.e. Monte Carlo, MCMC, etc.) should be maintained in the corresponding directories.
