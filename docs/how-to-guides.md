## How to construct a "predictive" engineering model (PEM)
1. First, break the system into individual components, for example the cathode, the discharge channel, the far-field plume, etc. Define all the inputs and outputs of each component and how they connect to each other. Variables that get mapped from the output of one component into the input of another component are termed "coupling" variables, while all other inputs are called system "exogenous" variables.
2. Write a Python wrapper function for each component model, following the [`amisc` package guidelines](https://eckelsjd.github.io/amisc/how-to-guides/). See examples for the [cathode](https://github.com/JANUS-Institute/HallThrusterPEM/blob/dcb0819c04a62a8950cfaa0720ed090d2bb5cdce/src/hallmd/models/cc.py#L17), [thruster](https://github.com/JANUS-Institute/HallThrusterPEM/blob/dcb0819c04a62a8950cfaa0720ed090d2bb5cdce/src/hallmd/models/thruster.py#L161), and [plume](https://github.com/JANUS-Institute/HallThrusterPEM/blob/dcb0819c04a62a8950cfaa0720ed090d2bb5cdce/src/hallmd/models/plume.py#L30) models. By convention, these functions are placed in the `hallmd.models` subdirectory.
3. Construct an `amisc.SystemSurrogate` object that links all the component models in a multidisciplinary system. See the example for [pem_v0](https://github.com/JANUS-Institute/HallThrusterPEM/blob/dcb0819c04a62a8950cfaa0720ed090d2bb5cdce/src/hallmd/models/pem.py#L31). Variables are specified in and loaded from a `.json` configuration file in the `hallmd.models.config` directory. Other model configurations can also go here.
4. You can test your system using the `SystemSurrogate` object.

!!! Example
    ```python
    from hallmd.models.pem import your_pem
    
    pem_obj = your_pem()
    inputs  = pem_obj.sample_inputs(100)                 # (100, input_dim)
    outputs = pem_obj.predict(inputs, use_model='best')  # (100, output_dim)
    ```
    This will evaluate all the component models directly using the "best" available model, which will be the highest fidelity specified in your wrapper functions. It typically won't be feasible to do this with a large number of inputs, so next you will want to train the `pem_obj` surrogate to approximate the true system model.

## How to train a surrogate for the PEM
This guide assumes you have access to a Linux HPC system. It will use specific examples for the Great Lakes system at the University of Michigan, which uses the `SLURM` workload manager and the `Lmod` environment module. Regardless, this guide can be adapted to your specific system as needed. We assume you have a terminal connection open on the system you are using.

1. Clone this repository and change into the root project directory:
```shell
git clone https://github.com/JANUS-Institute/HallThrusterPEM.git
cd HallThrusterPEM
```
2. Source the setup script in the shell:
```shell
source setup_env.sh
```
This will make sure you have the `pdm` tool installed, a proper version of Python loaded, the `mpi4py` library installed, and all required SLURM environment variables defined. You will need to edit this script for your specific usage, for example adding your SLURM account info, commenting out `pdm add mpi4py` if you do not have an MPI-enabled system, etc. The main idea here is to set up a working Python virtual environment with all resources defined and loaded.
3. Create a new directory in the `scripts` folder with the name of your new "PEM" system. The `scripts/pem_v0` contains everything used to build the original 3-component PEM; it may be easiest to simply copy this directory as a template and name it `pem_vi` for $i > 0$.
4. Look at the `[tool.pdm.scripts]` section of `pyproject.toml`. There are three convenience scripts provided that can be run with the command `pdm run script_name`: the important ones are `gen_data`, `fit`, and `train`.
5. The `gen_data` script will call `sbatch scripts/your_pem/gen_data.sh` which then calls `gen_data.py`. This is responsible for generating all the data needed by the models and surrogate before training the surrogate. For example, `pem_v0` relies on `gen_data` to make a test set and some compression-related data that get copied over to the `hallmd.models.config` directory. At the very least, you will likely need this to make a test set for evaluating the performance of the surrogate during training.
6. The `fit` script will call `sbatch scripts/your_pem/fit_surr.sh` which then calls `fit_surr.py`. This is responsible for actually loading your `SystemSurrogate` object and training the surrogate via:
```python
from hallmd.models.pem import your_pem

pem_obj = your_pem()
your_pem.fit(max_iter=100, max_runtime_hr=3)  # for example
```
7. The `train` script is an expedient for calling `gen_data` and `fit` in sequence, with the latter being dependent on the successful completion of the first.

!!! Example "TLDR; Complete working example"
    ```shell
    git clone https://github.com/JANUS-Institute/HallThrusterPEM.git
    cd HallThrusterPEM
    
    # Make your specific edits to setup_env.sh
    
    source setup_env.sh
    
    # Make a scripts/pem_v1 folder and edit the gen_data, fit_surr, etc. files
    
    pdm train pem_v1
    ```

## How to use the surrogate after training
Surrogate training is performed with the `amisc.SystemSurrogate.fit` function. You should specify a save directory for the `SystemSurrogate` object, which will create a folder with the hierarchy:
```shell
amisc_2024_timestamp     # Root surrogate directory
|- components            # Model output files may optionally be saved here
|  |- Cathode
|  |- Thruster
|  |- etc.
|- sys                   # Surrogate save files
|  |- sys_init.pkl
|  |- etc.
|  |- sys_final.pkl
|- 2024_timestamp.log    # Training log (useful for debugging)
```
You can freely distribute the standalone `.pkl` save files and reload the surrogate using the `load_from_file()` function:
```python
from amisc.system import SystemSurrogate

file = 'sys_final.pkl'
surr = SystemSurrogate.load_from_file(file)
```

!!! Note 
    It is more advisable to distribute the whole `amisc_timestamp` directory and load the save file from within `amisc_timestamp/sys/sys_final.pkl`, since the directory structure will be recreated from a standalone file regardless.

## How to use the surrogate for uncertainty quantification
There are four more scripts provided in `scripts/pem_v0` that were used to run all UQ analyses for the original 3-component PEM:

1. `plot_slice.py` -- Loads the surrogate from a training save `.pkl` file and plots several "1d slices" of inputs and outputs and compares to the true model output. This is useful for gauging how good the surrogate approximation is.
2. `mcmc.py` -- Contains several functions for maximum-likelihood estimation, obtaining a Laplace estimate of the posterior, and Markov-Chain Monte Carlo (MCMC) sampling using the `uqtils` package. Use this for calibration of the PEM model parameters using the surrogate.
3. `monte_carlo.py` -- Samples the uncertain inputs and propagates through the surrogate to get output uncertainty. Has several plotting functions for comparing model predictions to experimental data.
4. `sobol.py` -- Performs a Sobol' sensitivity analysis using the surrogate and the `uqtils` package. Has several plotting functions for showing Sobol' indices for each component model.

!!! Note
    It is advisable to copy all these files from `pem_v0` and adapt them for your new `pem_vXX`. They are written quite specific to the use case, but can serve as a good starting point for your own scripts.