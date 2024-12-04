This directory is used for running UQ analysis on different PEM configurations and experimental data.

## Predictive engineering models (PEM)

We define a "PEM" as a set of coupled component models that predicts global performance metrics and quantities of interest (QoIs), such as thrust, ion velocity, and efficiency, as a function of global inputs, such as discharge voltage, mass flow rate, and background chamber pressure. 

We currently only define a single PEM configuration:

- **PEM v0** - a purely feedforward coupling between a semi-empirical cathode model, a 1d fluid thruster model, and a semi-empirical plume model. See the corresponding directory for analysis scripts.

A PEM must define the sets of parameters and ranges over which it seeks to be applicable. Any significant change to the parameters, underlying models, or the addition of new models or parameters would warrant a new "pem_vXX" directory for analysis. The underlying models themselves (rather than the specific parameters or configurations) should be maintained in the `src/hallmd` Python package. The `hallmd` package also maintains devices, experimental data, and loading functions for each.

## Scripts
Below are several general scripts that can be used for any PEM configuration:

### Setup on HPC
Installs `pdm` and the `hallmd` package. Also installs Julia and `HallThruster.jl`.
```shell
./setup_hpc.sh --hallthruster-version=0.17.2 --julia-version=1.10
```

### Train a surrogate
Loads a surrogate configuration from file, generates compression and test set data, and trains the surrogate.
```shell
./train_hpc.sh pem_v0/pem_v0_SPT-100.yml --compression_samples=100 --test_samples=100 --max_iter=100 --executor=thread
```

You can also use `./train_local.sh` to train the surrogate locally rather than submitting HPC jobs.

### Plot surrogate slices
Loads a trained surrogate from file and plots slices over the input space against the true model.
```shell
./slice_hpc.sh pem_v0 --executor=thread
```

This will look for the most recent save file in the `pem_v0` directory (but you can also directly specify a save file).

## PEM analysis procedure
For a new PEM configuration, follow this procedure for analysis:

1. Define a single `.yml` configuration file that specifies all models, parameters, configurations, etc. The format of this file follows the [`amisc` interface](https://eckelsjd.github.io/amisc/guides/config_file/). See also the config file for [PEM v0](pem_v0/pem_v0_SPT-100.yml) for an example.
1. Generate compression data for field quantities and a random test set for evaluating surrogate performance. See the [`gen_data.py`](gen_data.py) script.
1. Train a surrogate to approximate the model. See the [`fit_surr.py`](fit_surr.py) script (`gen_data` and `fit_surr` are also combined into one step using [`train_hpc.sh`](train_hpc.sh)).
1. Evaluate the performance of the surrogate. For example, you can [plot 1d slices](plot_slice.py) over the inputs and compare to the true model. Generally, < 10\% error is advisable before moving forward.
1. Calibrate the model parameters to fit experimental data using the surrogate as a cheap approximation. See the [MCMC calibration](pem_v0/mcmc.py) of PEM v0.
1. Run any other desired UQ analyses (Monte Carlo, Sobol' sensitivity analysis, etc.). The [PEM v0 directory](pem_v0) again provides a few examples.

In general, anytime _anything_ changes about the models (including parameter ranges, values, configurations, or general computations), then a new surrogate _must_ be built, including new compression data if applicable and a new test set. Note that individual component models can be modified and retrained independently of other components.
