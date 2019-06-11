import argparse
from itertools import islice
import json
from pathlib import Path
import warnings
from typing import Dict
import os

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning


def make_predicts(models, force=False):
    for model_path in models:
        print(model_path)
        model = "_".join(model_path.split("_")[1:-2])
        if not force and os.path.isfile(os.path.join(model_path, 'test.h5')):
            continue
        print(f'predicting {model_path} {model}')
        os.system(f'python -m imet.main predict_test {model_path} --model {model}')


def make_valid_predicts(models, force=False):
    for model in models:
        for fold in range(5):
            print(model, fold)
            model_path = f'zoo/model_{model}_fold_{fold}'
            if not force and os.path.isfile(os.path.join(model_path, 'val.h5')):
                continue
            print(f'predicting {model_path} {model}')
            os.system(f'python -m imet.main predict_valid {model_path} --model {model} --fold {fold} --input-size 320 --tta 16')        


def make_submission(models):
    models = ' '.join(map(lambda model_path: os.path.join(model_path, 'test.h5'), models))
    os.system(f'python -m imet.make_submission {models} submission_newensemble.csv --threshold 0.08')


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('models', nargs='+')        
    args = parser.parse_args()
    make_submission([f'zoo/model_{model}_fold_{fold}' for fold in range(5) for model in args.models])


if __name__ == '__main__':
    main()
