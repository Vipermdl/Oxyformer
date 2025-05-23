import math
import torch, zero
import torch.nn as nn
import typing as ty
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import sys
sys.path.append('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/')

from lib.models.model_utils import initialize_weights
from lib.models.model_utils import get_activation_fn, get_nonglu_activation_fn

from pathlib import Path
from lib.utils import load_config, get_path, dump_pickle, to_tensors, load_json, dump_stats
from lib.utils import TRAIN, TEST, VAL, format_seconds, backup_output
from lib.models.model_utils import get_device, get_n_parameters, make_optimizer
from lib.models.model_utils import IndexLoader, get_lr, is_oom_exception, train_with_auto_virtual_batch
from lib.models.transformer import Transformer

from lib.metric import make_summary, calculate_metrics
from lib.dataset import Dataset, normalize
from lib.loss import MSELoss

import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# from evaluate import usecols, labelcols

# from main import get_norm_func, evaluate_test_deep

class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num):
        x = []
        if x_num is not None:
            x.append(x_num)
        
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x



if __name__ == '__main__':
    args, output = load_config()

    zero.set_randomness(args['seed'])
    dataset_dir = get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = Dataset.from_dir(dataset_dir)

    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)
    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))

    dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else to_tensors(x) for x in X)

    Y = to_tensors(Y)   

    Y = {k: v.squeeze() for k, v in Y.items()} 

    device = get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, _ = X
    del X
    
    Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = MSELoss()

    # import pdb; pdb.set_trace()

    model = MLP(
        d_in=X_num['train'].shape[1],
        d_out=D.info['n_classes'],
        **args['model'],
    ).to(device)
    
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        args['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {TRAIN: [], VAL: [], TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    def apply_model(part, idx):
        return model(X_num[part][idx])

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in IndexLoader(
                                    D.size(part), eval_batch_size, False, device
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                    stats['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics[part] = calculate_metrics(
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        dump_stats(stats, output, final)
        backup_output(output)

    timer.run()

    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            loss, new_chunk_size = train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(TRAIN, x), Y_device[TRAIN][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                stats['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[TRAIN].extend(epoch_losses)
        print(f'[{TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([VAL])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])

    stats['metrics'], predictions = evaluate([VAL])

    # truth = Y['val'] * y_info['std'] + y_info['mean']
    # predictions = predictions['val'] * y_info['std'] + y_info['mean']

    truth = Y['val'] * (y_info['max'] - y_info['min']) + y_info['min']
    predictions = predictions['val'] * (y_info['max'] - y_info['min']) + y_info['min']

    mask = ~truth.isnan()

    truth = truth[mask]
    predictions = predictions[mask]

    stats['metrics']['R2'] = r2_score(truth, predictions)

    print("Val dataset R2 is: ", stats['metrics']['R2'])

    # import pdb; pdb.set_trace()
    policy = args['data'].get('y_policy')
    path = args['data']['path']
    norm = args['data'].get('normalization')
    seed = args['seed']
    r2, rmse, mae = evaluate_test_deep(policy=policy, path=path, model=model, norm=norm, seed=seed)

    stats['test'] = {'R2':r2, 'RMSE': rmse, 'MAE': mae}

    stats['time'] = format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
    
