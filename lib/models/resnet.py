import math
import torch, zero
import torch.nn as nn
import typing as ty
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import sys
sys.path.append('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/')

from pathlib import Path
from lib.utils import load_config, get_path, dump_pickle, to_tensors, load_json, dump_stats
from lib.utils import TRAIN, TEST, VAL, format_seconds, backup_output
from lib.models.model_utils import get_device, get_n_parameters, make_optimizer
from lib.models.model_utils import IndexLoader, get_lr, is_oom_exception, train_with_auto_virtual_batch

from lib.metric import make_summary, calculate_metrics
from lib.dataset import Dataset, normalize
from lib.loss import MSELoss

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from main import evaluate_test_deep




def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                     padding=kernel_size//2, bias=True)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Tokenizer(nn.Module):

    def __init__(
            self,
            d_numerical: int,
            d_token: int,
            bias: bool,
    ) -> None:
        super().__init__()

        d_bias = d_numerical
        self.category_offsets = None
        self.category_embeddings = None

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        x_some = x_num
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]

        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, d_numerical, d_token=32, token_bias=True, d_out=75):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.tokenizer = Tokenizer(d_numerical, d_token, token_bias)
        self.conv1 = nn.Conv1d(d_numerical+1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32, layers[0], kernel_size=1, stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], kernel_size=5, stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], kernel_size=5, stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], kernel_size=1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((256 * block.expansion, 1))
        self.fc = nn.Linear(256 * block.expansion, d_out)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size,
                            stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x, bounds=None):
        x = self.tokenizer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)

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

    model = ResNet(
        d_numerical=X_num['train'].shape[1],
        d_out=D.info['n_classes'],
        block = BasicBlock, layers=[2, 2, 2, 2]#**args['model'],
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

    import pdb; pdb.set_trace()
    policy = args['data'].get('y_policy')
    path = args['data']['path']
    norm = args['data'].get('normalization')
    seed = args['seed']
    r2, rmse, mae = evaluate_test_deep(policy=policy, path=path, model=model, norm=norm, seed=seed)

    stats['test'] = {'R2':r2, 'RMSE': rmse, 'MAE': mae}

    stats['time'] = format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
    
    
    
    
    

