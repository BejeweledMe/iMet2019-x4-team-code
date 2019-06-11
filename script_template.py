import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


models = ['efficientnet-b3', 'se_resnext101_32x4d', 'se_resnext50_32x4d']
params = {
    'se_resnext50_32x4d': {
        'input_size': 320,
        'folds': [0, 1, 4],
        # 0.596, 0.598, 0.593, 0.591, 0.596
        'tta': 6,
        'batch_size': 64,
        'source': '/kaggle/input/imetzoo/zoo/'
    },
    'se_resnext101_32x4d': {
        'input_size': 320,
        'folds': [1, 3, 4],
        # 0.595, 0.598, 0.596, 0.598, 0.598
        'tta': 6,
        'batch_size': 64,
        'source': '/kaggle/input/imetzoo/zoo/'
    },
    'senet154': {
        'input_size': 320,
        'folds': [0, 1, 4],
        'tta': 6,
        'batch_size': 64,
        'source': '/kaggle/input/imetzoo/zoo/'
    },
	'efficientnet-b3': {
        'input_size': 384,
        'folds': [0, 1, 4],
		# 0.549. 0.552, 0.546, 0.545, 0.549
        'tta': 6,
        'batch_size': 64,
        'source': '/kaggle/input/efficientnet/efficient/efficient/'
	}
}

run('python setup.py develop --install-dir /kaggle/working')
run('python -m imet.make_folds')
for model in models:
    for fold in params[model]['folds']:
        run_root = params[model]['source']
        input_size = params[model]['input_size']
        tta = params[model]['tta']
        batch_size = params[model]['batch_size']
        run(f'python -m imet.main predict_test {run_root}model_{model}_fold_{fold} --model {model} --pretrained 0 --fold {fold} --input-size {input_size} --batch-size {batch_size} --tta {tta}')
        
predictions = []

for model in models:
    for fold in params[model]['folds']:
        predictions.append(f'/kaggle/working/test_model_{model}_fold_{fold}.h5')

predictions = ' '.join(predictions)

run(f'python -m imet.make_submission {predictions} submission.csv --threshold 0.08')