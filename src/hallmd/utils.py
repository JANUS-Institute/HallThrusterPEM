"""Module to provide utilities for the `hallmd` package.

Includes:
"""
import json
import os
from pathlib import Path

import yaml


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


def load_thruster(thruster_dir: str | Path, thruster_filename: str = 'thruster.yml') -> dict:
    """Load a device configuration from the `device_dir` directory. The `device_file` must be located at
    `device_dir/device_name/device_file`. All other files in the directory, if referenced in `device_file`, will
    be converted to an absolute path.

    !!! Example "Loading a device configuration"
        Currently, the only provided device configuration is for the SPT-100 thruster.
        ```python
        from hallmd.utils import load_thruster

        device = load_thruster('devices/SPT-100')
        ```

    The format of a device file is as follows
    ```yaml
    name: MyDevice
    geometry:
      channel_length: 1
      inner_radius: 2
      outer_radius: 3
    magnetic_field:
      file: bfield.csv
    shielded: false
    ```

    :param device_name: name of the device configuration to load
    :param device_file: name of the device configuration file (default: 'device.yml'). Only supported file types are
                        `.yml` and `.json`.
    :param device_dir: directory containing the devices. If None, the `hallmd.devices` directory is used.
    :return: dictionary containing the device configuration
    """
    thruster_file = Path(thruster_dir) / thruster_filename

    with open(thruster_file, 'r', encoding='utf-8') as fd:
        if thruster_file.suffix == '.yml':
            config = yaml.safe_load(fd)
        elif thruster_file.suffix == '.json':
            config = json.load(fd)
        else:
            raise ValueError(
                f'Unsupported file type "{thruster_file.suffix}". Only .yml and .json files are supported.'
            )

    # Convert all relative paths to absolute paths for loading bfields, etc
    for root, _, files in os.walk(thruster_dir):
        for file in files:
            if file != thruster_file:
                # Check if the posix file path from root is in the config (i.e. "./file.csv")
                root_path = Path(root) / file  # Path like "hallmd/devices/SPT-100/path/to/file.csv"
                rel_path = root_path.relative_to(thruster_dir)  # Just the path/to/file.csv part (relative)
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
