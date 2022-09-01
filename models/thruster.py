# Module for thruster models

from juliacall import Main as jl
import logging
import sys
from pathlib import Path

sys.path.append('..')
logger = logging.getLogger(__name__)
jl.seval("using HallThruster")

from utils import data_load, data_write


def hall_thruster_jl_model(thruster_input='thruster_input.json', cc_input=None):
    logger.info('Running thruster model')
    data_dir = Path('../data')

    # Make sure correct B-field filename
    input_data = data_load(thruster_input)
    b_field_file = input_data['design']['magnetic_field_file']
    b_field_fullfile = str(data_dir / b_field_file)
    input_data['design']['magnetic_field_file'] = b_field_fullfile

    # Load cc_input
    V_cc = input_data['design']['cathode_potential']
    if cc_input:
        cc_data = data_load(cc_input)
        input_data['design']['cathode_potential'] = cc_data['V_cc']

    data_write(input_data, thruster_input)

    # Call Hallthruster.jl simulation
    sol = jl.HallThruster.run_simulation(str(data_dir / thruster_input))
    jl.HallThruster.write_to_json(str(data_dir / 'thruster_output.json'), jl.HallThruster.time_average(sol))

    # Restore the input.json file
    input_data['design']['magnetic_field_file'] = b_field_file
    input_data['design']['cathode_potential'] = V_cc
    data_write(input_data, thruster_input)

    return sol
