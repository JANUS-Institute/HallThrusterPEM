import os
import json
from pathlib import Path
import copy


def main():
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
    main()
