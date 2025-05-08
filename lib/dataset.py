import dataclasses as dc
import pickle, os
import typing as ty
import warnings
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.preprocessing
import torch

# import lib.utils as util
from lib.utils import usecols, labelcols, raise_unknown, load_json, DATA_DIR
    

ArrayDict = ty.Dict[str, np.ndarray]


def get_norm_func(train_dir, file_name, args):
    train_dir = Path(train_dir)
    df_train = pd.read_csv(train_dir / file_name)
    # df_train = set_position_sin_cos(df_train)
    df_train = {'train': df_train[usecols].values}
    _, norm_func = normalize(df_train, 
                normalization=args['data'].get('normalization'),
                seed=args['seed'])
    return norm_func


def normalize(
    X: ArrayDict, normalization: str, seed: int, noise: float = 1e-3
) -> ArrayDict:
    X_train = X['train'].copy()
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    else:
        # util.raise_unknown('normalization', normalization)
        raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}, normalizer  # type: ignore[code]


@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]
    folder: ty.Optional[Path]

    @classmethod
    def from_dir(cls, dir_: ty.Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)

        def load(item):
            return {
                x: ty.cast(np.ndarray, temp) for x, temp in zip(['train', 'val'], item)
            }

        df_train = pd.read_csv(dir_ / 'train.csv')
        df_test = pd.read_csv(dir_ / 'val.csv')
        
        df_train = df_train[~np.isnan(df_train[labelcols]).all(axis=1)]
        df_train.reset_index(drop=True, inplace=True)
        df_test = df_test[~np.isnan(df_test[labelcols]).all(axis=1)]
        df_test.reset_index(drop=True, inplace=True)
        
        info = {'n_num_features':len(usecols), 'n_classes':len(labelcols)} #43 
        train, train_label = df_train[usecols], df_train[labelcols[:info['n_classes']]]
        test, test_label = df_test[usecols], df_test[labelcols[:info['n_classes']]]  

        return Dataset(
            load([train.values, test.values]),
            load([train_label.values, test_label.values]),
            info,
            dir_,
        )

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features

    def size(self, part: str) -> int:
        X = self.N
        assert X is not None
        return len(X[part])

    def build_X(
        self,
        *,
        normalization: ty.Optional[str],
        num_nan_policy: str,
        seed: int,
    ) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        cache_path = (
            self.folder
            / f'build_X__{normalization}__{num_nan_policy}__{seed}.pickle'  # noqa
            if self.folder
            else None
        )
        
        if cache_path and cache_path.exists():
            print(f'Using cached X: {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        def save_result(x):
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(x, f)

        if self.N:
            N = deepcopy(self.N)
            if normalization:
                N, _ = normalize(N, normalization, seed)
        else:
            N = None

        assert N is not None
        # save_result(N)
        return N


    def build_y(
        self, policy: ty.Optional[str]
    ) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        
        # assert 
        y = deepcopy(self.y)
        
        if policy == 'mean_std':
            mean, std = [], []
            for index in range(self.y['train'].shape[1]):
                item = self.y['train'][:, index]
                mask = ~np.isnan(item)
                mean.append(item[mask].mean())
                std.append(item[mask].std())
            mean = np.array(mean).reshape([1, -1])
            std = np.array(std).reshape([1, -1])
            y = {k: (v - mean) / std for k, v in y.items()}
            info = {'policy': policy, 'mean': mean, 'std': std}
        elif policy == 'max_min':
            max, min = [], []
            for index in range(self.y['train'].shape[1]):
                item = self.y['train'][:, index]
                mask = ~np.isnan(item)
                max.append(item[mask].max() if len(item[mask]) > 1 else 1)
                min.append(item[mask].min() if len(item[mask]) > 1 else 0)
            max = np.array(max).reshape([1, -1])
            min = np.array(min).reshape([1, -1])
            y = {k: (v - min) / (max - min) for k, v in y.items()}
            info = {'policy': policy, 'max': max, 'min': min}
        
        return y, info


def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).float() for k, v in data.items()}


def load_dataset_info(dataset_name: str) -> ty.Dict[str, ty.Any]:
    # info = util.load_json(util.DATA_DIR / dataset_name / 'info.json')
    info = load_json(DATA_DIR / dataset_name / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    return info
