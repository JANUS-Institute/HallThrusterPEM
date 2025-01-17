"""Module to provide utilities for the `hallmd` package.

Includes:

- `load_device()` - Load a device configuration from the `hallmd.devices` directory.
- `data_write()` - Convenience function for writing .json data to file.
- `plot_qoi()` - Convenience plotting tool for showing QoI with UQ bounds
"""
import json
import os
from importlib import resources
from pathlib import Path

import yaml

__all__ = ['load_device']

AVOGADRO_CONSTANT = 6.02214076e23      # Avogadro constant (mol^-1)
FUNDAMENTAL_CHARGE = 1.602176634e-19   # Fundamental charge (C)
BOLTZMANN_CONSTANT = 1.380649e-23      # Boltzmann constant (J/K)
TORR_2_PA = 133.322                    # Conversion factor from Torr to Pa
MOLECULAR_WEIGHTS = {
    'Xenon': 131.293,
    'Argon': 39.948,
    'Krypton': 83.798,
    'Bismuth': 208.98,
    'Hydrogen': 1.008,
    'Mecury': 200.59
}


def _path_in_dict(value, data: dict) -> list:
    """Recursively check if a value is in a dictionary and return the list of access keys to the value."""
    if isinstance(data, dict):
        for key, v in data.items():
            path = _path_in_dict(value, v)
            if path:
                return [key] + path
    elif data == value:
        return [value]  # found the value
    return []           # value not found


def load_device(device_name: str, device_file: str = 'device.yml', device_dir: str | Path = None) -> dict:
    """Load a device configuration from the `device_dir` directory. The `device_file` must be located at
    `device_dir/device_name/device_file`. All other files in the directory, if referenced in `device_file`, will
    be converted to an absolute path.

    !!! Example "Loading a device configuration"
    Currently, the only provided device configuration is for the SPT-100 thruster.
    ```python
    from hallmd.utils import load_device

    device = load_device('SPT-100')
    ```
    You may put custom configurations in the `hallmd.devices` directory or specify a custom directory with a custom
    configuration file:
    ```yaml
    name: MyDevice
    geometry:
      channel_length: 1
      inner_radius: 2
      outer_radius: 3
    magnetic_field: bfield.csv
    shielded: false
    ```

    :param device_name: name of the device configuration to load
    :param device_file: name of the device configuration file (default: 'device.yml'). Only supported file types are
                        `.yml` and `.json`.
    :param device_dir: directory containing the devices. If None, the `hallmd.devices` directory is used.
    :return: dictionary containing the device configuration
    """
    device_dir = resources.files('hallmd.devices') if device_dir is None else Path(device_dir)
    if not (device_dir / device_name).exists():
        raise FileNotFoundError(f'Device directory "{device_name}" not found in the device folder.')
    if not (device_dir / device_name / device_file).exists():
        raise FileNotFoundError(f'Device configuration file "{device_file}" not found in the "{device_name}" '
                                f'directory. Please rename or specify the configuration file as "{device_file}".')

    config_file = device_dir / device_name / device_file
    with open(config_file, 'r', encoding='utf-8') as fd:
        if config_file.suffix == '.yml':
            config = yaml.safe_load(fd)
        elif config_file.suffix == '.json':
            config = json.load(fd)
        else:
            raise ValueError(f'Unsupported file type "{config_file.suffix}". Only .yml and .json files are supported.')

    # Convert all relative paths to absolute paths
    for root, _, files in os.walk(device_dir / device_name):
        for file in files:
            if file != device_file:
                # Check if the posix file path from root is in the config (i.e. "./file.csv")
                root_path = Path(root) / file  # Path like "hallmd/devices/SPT-100/path/to/file.csv"
                rel_path = root_path.relative_to(device_dir / device_name)  # Just the path/to/file.csv part (relative)
                dict_path = _path_in_dict(rel_path.as_posix(), config)
                if len(dict_path) == 0:
                    # Check if the plain filename is in the config (i.e. file.csv); will only pick first match
                    dict_path = _path_in_dict(file, config)

                if dict_path:
                    d = config  # pointer to the nested location in config
                    for key in dict_path[:-2]:
                        d = config[key]
                    d[dict_path[-2]] = root_path.resolve().as_posix()

    return config
