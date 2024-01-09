import os
import yaml
from functools import partial

logs_path = 'log_train/train'

versions = os.listdir(logs_path)

with open('params.csv', 'w') as f_out:
    head_list = []

    param_list = []
    for version in versions:
        params_file = os.path.join(logs_path, version, 'hparams.yaml')
        with open(params_file, 'r') as f:
            lines = f.readlines()
            yaml_stream = lines[:10]
            yaml_stream += lines[18:]
            stream = ''
            for line in yaml_stream:
                stream += line + '\n'
            params = yaml.safe_load(stream)
            param_list.append(params)

    for params in param_list:
        for k in params.keys():
            if k not in head_list:
                head_list.append(k)

    f_out.write('version,')
    for k in head_list:
        f_out.write(k + ',')
    f_out.write('\n')

    for params, version in zip(param_list, versions):
        f_out.write(version + ',')
        for k in head_list:
            if k not in params.keys():
                f_out.write(',')
            else:
                f_out.write(f"{params[k]},")
        f_out.write('\n')
