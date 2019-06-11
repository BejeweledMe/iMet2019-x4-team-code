import numpy as np
import pandas as pd

DATA_PATH = '/data/train/'

train = pd.read_csv(DATA_PATH + 'train.csv')
# train_ext = pd.read_csv(DATA_PATH + 'train_ext.csv')
test = pd.read_csv('submission_ensemble_all.csv')

files = [
    # '__kinoa__/2019-01-01_18-45-28_Inf_0/test_preds_inf0_0.csv',
    # '__kinoa__/2019-01-01_18-56-29_Inf_1/test_preds_inf1_0.csv',
    # '__kinoa__/2019-01-01_18-56-40_Inf_0/test_preds_inf0_0.csv',
    # '__kinoa__/2019-01-01_20-23-21_Inf_0/test_preds_inf0_0.csv',
    # '__kinoa__/2019-01-01_20-23-57_Inf_1/test_preds_inf1_0.csv',
    # '__kinoa__/2019-01-01_20-33-49_Inf_0/test_preds_inf0_0.csv',

    # '__kinoa__/2019-01-02_23-54-48_Inf_0/test_preds_inf0_0.csv',
    # '__kinoa__/2019-01-03_00-38-15_Inf_0/test_preds_inf0_0.csv',
    '__kinoa__/2019-01-03_10-45-00_Igor_preds/preds_ens-2.csv',
]

# Categorize target
y = np.zeros((train.shape[0], 28), dtype=int)
for i, t in enumerate(train['attribute_ids'].values):
    vals = str(t).split(' ')
    for v in vals:
        y[i, int(v)] = 1

# ye = np.zeros((train_ext.shape[0], 28), dtype=int)
# for i, t in enumerate(train_ext['attribute_ids'].values):
#     vals = str(t).split(' ')
#     for v in vals:
#         ye[i, int(v)] = 1

yt = np.zeros((test.shape[0], 28), dtype=int)
for i, t in enumerate(test['attribute_ids'].values):
    vals = str(t).split(' ')
    if len(vals) > 0 and vals[0] != 'nan':
        for v in vals:
            yt[i, int(v)] = 1

# Evaluate distributions
d_train = []
d_etrain = []
d_pred = []
for c_i in range(28):
    n_t = np.sum(y[:, c_i])
    n_tp = n_t/y.shape[0]
    d_train.append(n_tp)

    n_te = np.sum(ye[:, c_i])
    n_tep = n_te/ye.shape[0]
    d_etrain.append(n_tep)

    n_tt = np.sum(yt[:, c_i])
    n_ttp = n_tt/yt.shape[0]
    d_pred.append(n_tep)


d_test = [
    0.36239782, # 0
    0.043841336,
    0.075268817, # 2
    0.059322034,
    0.075268817, # 4
    0.075268817,
    0.043841336, # 6
    0.075268817,
    0, # 8
    0, # 9
    0, # 10
    0.043841336, # 11
    0.043841336, # 12
    0.014198783,
    0.043841336, # 14
    0, # 15
    0.014198783, # 16
    0.014198783,
    0.028806584,# 18
    0.059322034,
    0, # 20
    0.126126126,
    0.028806584,
    0.075268817,
    0, # 24
    0.222493888,
    0.028806584,
    0, # 27
]

for i in range(28):
    if d_test[i] == 0:
        d_test[i] = d_train[i]

print('C\tTrain\t\tExtra\t\tTest\t\tPredicted')
for c_i in range(28):
    print('{}:\t{:.9f},\t{:.9f},\t{:.9f},\t{:.9f}'.format(c_i, \
        d_train[c_i], d_etrain[c_i], d_test[c_i], d_pred[c_i]))
print('T:\t{:.9f},\t{:.9f},\t{:.9f},\t{:.9f}'.format( \
        np.sum(d_train), np.sum(d_etrain), np.sum(d_test), np.sum(d_pred)))

col_id = 'id'
preds = []
for f_i, file in enumerate(files):
    print(f_i, file)
    df = pd.read_csv(file)
    df = df.sort_values(col_id)
    # print(df)
    ids = df[col_id].values
    cols = sorted(df.columns)
    cols.remove(col_id)
    classes = [int(c.split('_')[-1]) for c in cols]
    cols = np.array(cols)[np.argsort(classes)]
    print(cols)

    p = df[cols].values 
    preds.append(p)

n_preds = len(preds)
preds = np.mean(preds, axis=0)

for c_i in range(28):
    idx = np.argsort(preds[:, c_i])[::-1]
    p = np.zeros(preds.shape[0])
    # print(int(preds.shape[0] * d_test[c_i]))
    p[idx[:int(preds.shape[0] * d_test[c_i])]] = 1
    # print(np.sum(p))
    preds[:, c_i] = p
    # print(np.sum(preds[:, i]))


test_full_preds = np.array(preds, dtype=int)
print(test_full_preds, np.sum(test_full_preds))
labels = []
for p in test_full_preds:
    s = []
    for i in range(test_full_preds.shape[1]):
        if p[i] > 0:
            s.append(str(i))

    labels.append(' '.join(s))

subm = pd.DataFrame({
    'Id': test['Id'],
    'Predicted': labels,
    })
print(subm.head())


subm.to_csv('submission_ensr.csv', index=False)

