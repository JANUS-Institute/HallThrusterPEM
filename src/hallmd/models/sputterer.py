""" `sputterer.py`

Module for Hall thruster sputtering models.

Includes
--------
- `sputterer_input()` - Used to format inputs for sputterer executable
- `sputterer_model()` - Used to run ./sputterer for the inputs
- `sputterer_wrapper()` - The main wrapper function that is compatible with `SystemSurrogate`
"""
from pathlib import Path
import time
import pickle
import copy
import json
import tempfile
import os
import random
import string
import toml
import csv
import subprocess

import numpy as np
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler
from amisc.utils import load_variables, get_logger

from hallmd.utils import ModelRunException, data_write, model_config_dir

Q_E = 1.602176634e-19   # Fundamental charge (C)
CONFIG_DIR = model_config_dir()


def sputterer_sim_input(sputterer_input: dict) -> dict:
    """Format inputs for the sputterer executable.

    :param sputterer_input: dictionary with all named sputterer inputs and values
    :returns: a toml file in the format that the sputterer executable expects
    """

    config = dict()
    config['simulation'] = sputterer_input['simulation']
    config['plume_model'] = sputterer_input['plume_model']
    config['chamber'] = sputterer_input['chamber']
    config['material'] = sputterer_input['material']
    config['geometry'] = sputterer_input['geometry']

    # with open('config.toml', 'w') as toml_file:
    #     toml.dump(config, toml_file)
    return config


def sputterer_model(sputterer_input: dict, executable_path: str) -> dict:
    """Run a sputterer simulation for given inputs.

    :param sputterer_input: named key-value pairs of sputterer inputs
    :raises ModelRunException: if anything fails in sputterer
    :returns: `dict` of sputterer outputs for this input
    """

    # Format inputs for sputterer
    config = sputterer_sim_input(sputterer_input)

    # Run simulation
    try:
        fd = tempfile.NamedTemporaryFile(suffix='.toml', encoding='utf-8', mode='w', delete=False)
        toml.dump(config, fd)
        fd.close()
        t1 = time.time()
        sol = subprocess.run([executable_path/'sputterer', fd.name, '0'], check=True,
                             capture_output=True)
        os.unlink(fd.name)   # delete the tempfile
    
    except subprocess.CalledProcessError as e:
        raise ModelRunException(f"An error occurred: {e.stderr}")
    else:
        print(f"Command output: {sol.stdout}, Retcode: {sol.returncode}")

    # Load simulation results
    with open(CONFIG_DIR / 'deposition.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        results = list(csv_reader)
    
    print(results)
    return results

if __name__ == '__main__':
    with open(Path(CONFIG_DIR / 'sputterer_input.toml'), 'r') as fd:
        config_data = toml.load(fd)
    sputterer_model(config_data, "/mnt/c/Users/gemte/Desktop/University/SummerResearch24/HallThrusterPEM/src/hallmd/models/sputterer")