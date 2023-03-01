# Combine sub-models

import numpy as np
import sys
import logging
import copy

sys.path.append('..')
logging.basicConfig(level=logging.INFO)

# Custom imports
from models.cc import cathode_coupling_model_feedforward as cc_model
from models.thruster import hall_thruster_jl_model as thruster_model
from models.plume import jion_modified as plume_model


def feedforward_pem(model_inputs, jl=None):
    """Run simple vcc-thruster-plume model"""
    assert len(model_inputs) == 3

    # Allocate space for return dictionary
    pem_result = {'pem_version': 'feedforward', 'cc': {}, 'thruster': {}, 'plume': {}}

    # Run cathode-coupling model
    cc_input = model_inputs[0]
    cc_output = cc_model(cc_input)
    pem_result['cc']['input'] = cc_input
    pem_result['cc']['output'] = cc_output

    # Run Hallthruster.jl model
    thruster_input = model_inputs[1]
    pem_result['thruster']['input'] = copy.deepcopy(thruster_input)
    thruster_input.update(cc_output)
    thruster_output = thruster_model(thruster_input, jl=jl)
    pem_result['thruster']['output'] = thruster_output

    # # Run plume model
    plume_input = model_inputs[2]
    pem_result['plume']['input'] = copy.deepcopy(plume_input)
    plume_input.update(thruster_output)
    plume_output = plume_model(plume_input)
    pem_result['plume']['output'] = plume_output

    return pem_result
