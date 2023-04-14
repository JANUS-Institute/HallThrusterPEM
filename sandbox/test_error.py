import os
import json
from pathlib import Path
import copy

from models.cc import cathode_coupling_model as cc_model
from models.thruster import hall_thruster_jl_model as thruster_model
from models.plume import current_density_model as plume_model
from utils import ModelRunException, data_write
import traceback


def rerun_model(result_json_filename):
    """Rerun a test case from a result json file"""
    with open(result_json_filename, 'r', encoding='utf-8') as fd:
        data = json.load(fd)
    basename = result_json_filename.split('.json')[0]

    # Load inputs
    model_inputs = []
    for model_iter, model_io in data.items():
        if model_iter != 'Exception':
            model_inputs.append(model_io['input'])

    # Copy inputs
    results = dict()
    for j, model_input in enumerate(model_inputs):
        results[f'model{j}'] = {'input': copy.deepcopy(model_input)}

    # Load models
    models = [cc_model, thruster_model, plume_model]

    try:
        # Run models
        for j, model in enumerate(models):
            model_out = model(model_inputs[j])
            results[f'model{j}']['output'] = copy.deepcopy(model_out)

            if j < len(models) - 1:
                model_inputs[j + 1].update(model_out)

    except ModelRunException as e:
        tb_str = traceback.format_exc()
        results['Exception'] = tb_str
        data_write(results, f'{basename}_retry_exc.json', dir='../test')
    else:
        data_write(results, f'{basename}_retry.json', dir='../test')


def test_feedforward_error():
    dir = Path('../results/feedforward_mc')
    files = os.listdir(dir)
    plume_res = {}
    thruster_res = {}
    for i, f in enumerate(files):
        if 'exc' in str(f):
            with open(dir/str(f), 'r') as fd:
                data = json.load(fd)
                if 'plume' in data.get('Exception'):
                    plume_res[f'case{i}'] = copy.deepcopy(data['model2']['input'])
                else:
                    thruster_res[f'case{i}'] = copy.deepcopy(data['model1']['input'])

    with open('thruster_failed_inputs.json', 'w', encoding='utf-8') as fd:
        json.dump(thruster_res, fd, ensure_ascii=False, indent=4)

    with open('plume_failed_inputs.json', 'w', encoding='utf-8') as fd:
        json.dump(plume_res, fd, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    test_feedforward_error()

    # Place whatever result file you want to test in test/ directory
    rerun_model('../results/failures/ff_mc_15_exc_1.22e-05.json')


# JUNK local error computation for MF,MD surrogate refinement step (see models.surrogates)
# theta = 0.5
# global_idx = node_obj['global_out']
# fk_j = y_j[:, global_idx]
# fk_jk = y_jk[:, global_idx]
# coarse_surr = node_obj['surrogate'].get_sub_surrogate((0,)*len(alpha), (0,)*len(beta))
# x_coarse = np.concatenate((xtest[:, node_obj['exo_in']], y_jk[:, node_obj['global_in']]), axis=1)
# fk_coarse = coarse_surr(x_coarse)
# delta_mu = np.abs(np.nanmean(fk_jk, axis=0, keepdims=True) -
#                   np.nanmean(fk_j, axis=0, keepdims=True)) / np.abs(fk_coarse)
# delta_sigma = np.sqrt(np.abs(np.nanvar(fk_jk, axis=0, keepdims=True) -
#                              np.nanvar(fk_j, axis=0, keepdims=True))) / np.abs(fk_coarse)
# delta_error = np.nanmax(theta * delta_mu + (1 - theta) * delta_sigma)
