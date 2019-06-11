import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm

from .models import get_model
from .dataset import TrainDataset, TTADataset, get_ids, DATA_ROOT, TEST_FOLDS_ROOT
from .transforms import train_transform, test_transform
from .utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE, set_models_path_env, seed_everything, 
    _reduce_loss, _make_mask, binarize_prediction, N_CLASSES, create_class_weight)
from .losses import loss_function
from .optimizers import optimizer


def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'validate_best', 'predict_valid', 'predict_test'])
    arg('run_root')
    arg('--model', default='resnet50')
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 6)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=3)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=8)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--loss', type=str, default='bce')
    arg('--input-size', type=int, default=288)
    arg('--optimizer', type=str, default='adam')
    arg('--use-weight', type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train')
    set_models_path_env(args.model)
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    use_cuda = cuda.is_available()
    pos_weight = None
    if args.use_weight:
        pos_weight = torch.Tensor(create_class_weight(DATA_ROOT))
    if use_cuda and pos_weight is not None:
        pos_weight = pos_weight.cuda()
    criterion = loss_function(args.loss, pos_weight)
    
    model = get_model(args.model, num_classes=N_CLASSES, pretrained=args.pretrained, input_size=args.input_size, use_cuda=use_cuda)
    fresh_params = list()
    if args.model.startswith('efficientnet'):
        fresh_params = list(model._fc.parameters())
    else:
        fresh_params = list(model._classifier.parameters())


    if args.mode == 'train':
        is_continue = False
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        if run_root.exists():
            is_continue = True
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(train_fold, train_transform(args.input_size))
        valid_loader = make_loader(valid_fold, test_transform(args.input_size))
        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda optimzer, params, lr: optimizer(optimizer, params, lr),
            use_cuda=use_cuda
        )

        if args.pretrained and not is_continue:
            train(params=fresh_params, n_epochs=1, **train_kwargs)
        model = get_model(args.model, num_classes=N_CLASSES, pretrained=args.pretrained, input_size=args.input_size)
        if use_cuda:
            model = model.cuda()
        train_kwargs['model'] = model
        train(params=model.parameters(), **train_kwargs)

    elif args.mode == 'validate':
        valid_loader = make_loader(valid_fold, test_transform(args.input_size))
        load_model(model, run_root / 'model.pt')
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                   use_cuda=use_cuda, model_name=args.model)

    elif args.mode == 'validate_best':
        valid_loader = make_loader(valid_fold, test_transform(args.input_size))
        load_model(model, run_root / 'best-model.pt')
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'), 
            use_cuda=use_cuda, model_name=args.model)	

    elif args.mode.startswith('predict'):
        load_model(model, run_root / 'best-model.pt')
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            use_cuda=use_cuda,
            workers=args.workers,
            input_size=args.input_size
        )
        if args.mode == 'predict_valid':
            predict(model, df=valid_fold, root=train_root,
                    out_path=run_root / 'val.h5',
                    **predict_kwargs)
        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / (
                'test_sample' if args.use_sample else 'test')
            ss = pd.read_csv(TEST_FOLDS_ROOT / 'sample_submission.csv')
            if args.use_sample:
                ss = ss[ss['id'].isin(set(get_ids(test_root)))]
            if args.limit:
                ss = ss[:args.limit]
            predict(model, df=ss, root=test_root,
                    out_path=Path('/kaggle/working') / f'test_model_{args.model}_fold_{args.fold}.h5',
                    **predict_kwargs)


def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, use_cuda: bool, 
            input_size: int):
    loader = DataLoader(
        dataset=TTADataset(root, df, test_transform(input_size), tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=4, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(args.optimizer, params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model-1.pt'
    best_model_path = run_root / 'best-model.pt'
    best_valid_loss = 0.0
    if model_path.exists():
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
    lr_changes = 0

    save = lambda ep, save_name: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(run_root / save_name))

    report_each = 10
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = _reduce_loss(criterion(outputs, targets))

                batch_size = inputs.size(0)
                (batch_size * loss).backward()

                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                # if i and i % report_each == 0:
                #     write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1, f'model-{epoch}.pt')
            valid_metrics = validation(model, criterion, valid_loader, use_cuda, args.model)
            write_event(log, step, epoch, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save(epoch + 1, 'best-model.pt')
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr_changes +=1
                if lr_changes > max_lr_changes:
                    break
                lr /= 5
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch

                optimizer = init_optimizer(args.optimizer, params, lr)
                
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch, 'model-interrupted.pt')
            print('done.')
            return False
    return True


def validation(
        model: nn.Module, criterion, valid_loader, use_cuda, model_name
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets) 

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.07, 0.08, 0.09, 0.10, 0.15]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics



if __name__ == '__main__':
    main()
