import os


model = 'se_resnext101_32x4d'
fold = 0
input_size = 320
batch_size = 24
n_epochs = 15

os.system('python -m imet.make_folds')
os.system(f'python -m imet.main train model_{model}_fold_{fold} --n-epochs {n_epochs} --model {model} --fold {fold} --input-size {input_size} --clean --batch-size {batch_size}')
os.system(f'python -m imet.main predict_test model_{model}_fold_{fold} --model {model} --tta 16 --input-size {input_size} --batch-size {batch_size}')
