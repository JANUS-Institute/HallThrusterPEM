"""Testing for package utilities."""
import os
from importlib import resources
from pathlib import Path

from hallmd.utils import _path_in_dict, load_device


def test_path_in_dict():
    """Check recursively searching a dictionary for a value."""
    # Test single level dictionary
    single_level_dict = {'a': 1, 'b': 2, 'c': 3}
    assert _path_in_dict(2, single_level_dict) == ['b', 2]
    assert _path_in_dict(4, single_level_dict) == []

    # Test nested dictionary
    nested_dict = {'a': {'b': {'c': 3}}, 'd': {'e': 4}, 'f': 'hello'}
    assert _path_in_dict(3, nested_dict) == ['a', 'b', 'c', 3]
    assert _path_in_dict(4, nested_dict) == ['d', 'e', 4]
    assert _path_in_dict(5, nested_dict) == []
    assert _path_in_dict('hello', nested_dict) == ['f', 'hello']

    # Test value in multiple locations
    multi_location_dict = {'a': 1, 'b': {'c': 1, 'd': 2}, 'e': {'f': 1}}
    assert _path_in_dict(1, multi_location_dict) == ['a', 1]  # First occurrence
    assert _path_in_dict(2, multi_location_dict) == ['b', 'd', 2]

    mixed_dict = {
        'a': 1.1,
        'b': 2.0,
        'c': None,
        'd': True,
        'e': 'string_value',
        'f': {'g': 'nested_string', 'h': 3.14},
        'i': [1, 2, 'list_string']
    }
    assert _path_in_dict('string_value', mixed_dict) == ['e', 'string_value']
    assert _path_in_dict('nested_string', mixed_dict) == ['f', 'g', 'nested_string']
    assert _path_in_dict(1.1, mixed_dict) == ['a', 1.1]
    assert _path_in_dict(2.0, mixed_dict) == ['b', 2.0]
    assert _path_in_dict(None, mixed_dict) == ['c', None]
    assert _path_in_dict(True, mixed_dict) == ['d', True]


def test_load_device(tmp_path):
    """Test loading a device configuration."""
    # Test loading SPT-100 device configuration
    device_dir = resources.files('hallmd.devices')
    config = load_device('SPT-100')
    abs_mag_file = Path(device_dir / 'SPT-100' / 'bfield_spt100.csv').resolve().as_posix()
    assert config == {'name': 'SPT-100',
                      'geometry': {'channel_length': 0.025, 'inner_radius': 0.035, 'outer_radius': 0.05},
                      'magnetic_field': {'file': abs_mag_file},
                      'shielded': False}

    # Test a nested device configuration
    new_device_dir = tmp_path / 'nested_device'
    os.mkdir(new_device_dir)
    with open(new_device_dir / 'my_device.yml', 'w', encoding='utf-8') as fd:
        fd.writelines(['hello: there\n', 'my_file: data/my_file.txt\n', 'other_info:\n',
                       '  other_file: other_file.txt\n'])
    os.mkdir(new_device_dir / 'data')
    with open(new_device_dir / 'data' / 'my_file.txt', 'w', encoding='utf-8') as fd:
        fd.write('my data')
    with open(new_device_dir / 'data' / 'other_file.txt', 'w', encoding='utf-8') as fd:
        fd.write('other data')

    config = load_device('nested_device', 'my_device.yml', tmp_path)
    abs_file = Path(new_device_dir / 'data' / 'my_file.txt').resolve().as_posix()
    abs_other_file = Path(new_device_dir / 'data' / 'other_file.txt').resolve().as_posix()

    assert config == {'hello': 'there', 'my_file': abs_file, 'other_info': {'other_file': abs_other_file}}
