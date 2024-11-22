This directory is used for running UQ analysis on different PEM configurations and experimental data.

## Analysis procedure

1. Define a single `.yml` configuration file that specifies all models, parameters, configurations, etc. The format of this file follows the [`amisc` interface](https://eckelsjd.github.io/amisc/guides/config_file/). See also the config file for [PEM v0](pem_v0/pem_v0_SPT-100.yml) for an example.
1. Generate compression data for field quantities and a random test set for evaluating surrogate performance. See the [`gen_data.py`](gen_data.py) script.
1. Train a surrogate to approximate the model. See the [`fit_surr.py`](fit_surr.py) script.
1. Evaluate the performance of the surrogate. For example, you can [plot 1d slices](plot_slice.py) over the inputs and compare to the true model. Generally, < 10\% error is advisable before moving forward.
1. Calibrate the model parameters to fit experimental data using the surrogate as a cheap approximation. See the [MCMC calibration](pem_v0/mcmc.py) of PEM v0.
1. Run any other desired UQ analyses (Monte Carlo, Sobol' sensitivity analysis, etc.). The [PEM v0 directory](pem_v0) again provides a few examples.

In general, anytime _anything_ changes about the models (including parameter ranges, values, configurations, or general computations), then a new surrogate _must_ be built, including new compression data if applicable and a new test set. Note that individual component models can be modified and retrained independently of other components.

## Predictive engineering models (PEM)

We define a "PEM" as a set of coupled component models that predicts global performance metrics and quantities of interest (QoIs), such as thrust, ion velocity, and efficiency, as a function of global inputs, such as discharge voltage, mass flow rate, and background chamber pressure. 

We currently only define a single PEM configuration:

- **PEM v0** - a purely feedforward coupling between a semi-empirical cathode model, a 1d fluid thruster model, and a semi-empirical plume model. See the corresponding directory for analysis scripts.

A PEM must define the sets of parameters and ranges over which it seeks to be applicable. Any significant change to the parameters, underlying models, or the addition of new models or parameters would warrant a new "pem_vXX" directory for analysis. The underlying models themselves (rather than the specific parameters or configurations) should be maintained in the `src/hallmd` Python package. The `hallmd` package also maintains devices and experimental data, and loading functions for each.
