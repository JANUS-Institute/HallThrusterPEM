import json
from pathlib import Path
import copy

INTERFACE_DIR = Path(__file__).parent / 'interface'
INPUT_DIR = Path(__file__).parent / 'input'


def data_load(filename):
    """Convenience function to load .json data files"""
    file = INPUT_DIR / filename if 'input' in filename else INTERFACE_DIR / filename
    with open(file, 'r') as fd:
        data = json.load(fd)

    if 'input' in filename and 'system' not in filename:
        load_system_inputs(data)

    return data


def data_write(data, filename):
    """Convenience function to write .json data files"""
    with open(INTERFACE_DIR / filename, 'w', encoding='utf-8') as fd:
        json.dump(data, fd, ensure_ascii=False, indent=4)


def load_system_inputs(input_data):
    """Overwrite input parameters in 'input_data' with system input values"""
    system_data = data_load('system_input.json')
    for input_type, input_dict in system_data.items():
        for input_param, sys_value in input_dict.items():
            if input_data[input_type].get(input_param):
                input_data[input_type][input_param] = copy.deepcopy(sys_value)


def parse_input_file(file):
    """Parse generic component input file"""
    nominal_input = {}
    input_uncertainty = {}
    if file == 'thruster_input.json':
        nominal_input, input_uncertainty = parse_thruster_input(file)
    else:
        input_data = data_load(file)
        for input_type, input_params in input_data.items():
            for param, param_value in input_params.items():
                nominal_input[param] = param_value['nominal']
                if param_value['uncertainty'] != 'none':
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}

    return nominal_input, input_uncertainty


def parse_thruster_input(file):
    """Helper function to parse thruster input"""
    thruster_data = data_load(file)
    thruster_input = {}
    input_uncertainty = {}

    # Loop over all inputs and parse into thruster_input and input_uncertainty
    for input_type, input_params in thruster_data.items():
        for param, param_value in input_params.items():
            if input_type == 'simulation':
                thruster_input[param] = param_value
            elif 'material' in param:
                # Handle specifying material properties
                for mat_prop, prop_value in param_value.items():
                    if mat_prop == 'name':
                        thruster_input[param] = prop_value
                    else:
                        if prop_value['uncertainty'] != 'none':
                            input_uncertainty[mat_prop] = {'uncertainty': prop_value['uncertainty'],
                                                           'value': prop_value['value']}
            elif param == 'magnetic_field':
                # Handle different ways to specify magnetic field profile
                thruster_input['magnetic_field_file'] = param_value['magnetic_field_file']
                if param_value['uncertainty'] != 'none':
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}
            else:
                # Handle all generic input parameters
                thruster_input[param] = param_value['nominal']
                if param_value['uncertainty'] != 'none':
                    input_uncertainty[param] = {'uncertainty': param_value['uncertainty'],
                                                'value': param_value['value']}
    return thruster_input, input_uncertainty
