import argparse

import pandas as pd

from .utils import mean_df, binarize_prediction
from .dataset import DATA_ROOT, TEST_FOLDS_ROOT


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('output')
    arg('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    sample_submission = pd.read_csv(
        TEST_FOLDS_ROOT / 'sample_submission.csv', index_col='id')
    dfs = []
    for prediction in args.predictions:
        print(prediction)
        df = pd.read_hdf(prediction, index_col='id')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
    df = pd.concat(dfs)
    df = mean_df(df)
    df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv(args.output, header=True)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
