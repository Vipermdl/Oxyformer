import argparse
from calendar import calendar
import datetime
import json
import os
import pickle
import random
import shutil
import sys
import time
import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np

import pandas as pd
import pynvml
import pytomlpp as toml
import torch
import xarray as xr



TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]


ArrayDict = ty.Dict[str, np.ndarray]

PROJECT_DIR = Path(os.environ['PWD']).absolute().resolve()
# PROJECT_DIR = Path("/home/leadmove/dongliang/rtdl-main/").absolute().resolve()

DATA_DIR = PROJECT_DIR / 'dataset' / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

usecols = ['chlor_a', 'sosaline', 'sohtc700', 'somxl010', 'sohefldo', 'sohtcbtm', 'somxl030', 'sohtc300', 'v10', 'u10', 'mslp', 'sst', 'mslh', 'msnswrfcs', 'wmb']

labelcols  = [0.5057600140571594, 1.5558550357818604,  2.667681932449341,
       3.8562800884246826, 5.1403608322143555,  6.543034076690674,
         8.09251880645752,  9.822750091552734, 11.773679733276367,
       13.991040229797363, 16.525320053100586, 19.429800033569336,
       22.757619857788086, 26.558300018310547,  30.87455940246582,
        35.74020004272461,  41.18001937866211, 47.211891174316406,
        53.85063934326172,  61.11283874511719,  69.02168273925781,
        77.61116027832031,  86.92942810058594,  97.04131317138672,
        108.0302963256836,              120.0,  133.0758056640625,
        147.4062042236328,  163.1645050048828, 180.54989624023438,
        199.7899932861328, 221.14120483398438, 244.89059448242188,
        271.3564147949219, 300.88751220703125,    333.86279296875,
        370.6885070800781,  411.7939147949219,  457.6256103515625,
         508.639892578125,  565.2922973632812,  628.0260009765625,
        697.2587280273438,  773.3682861328125,  856.6790161132812,
        947.4478759765625,   1045.85400390625,  1151.990966796875,
       1265.8609619140625,     1387.376953125,  1516.364013671875,
       1652.5679931640625, 1795.6710205078125, 1945.2960205078125,
        2101.027099609375,  2262.422119140625,   2429.02490234375,
          2600.3798828125,       2776.0390625,  2955.570068359375,
         3138.56494140625,   3324.64111328125,  3513.446044921875,
        3704.656982421875,   3897.98193359375,  4093.158935546875,
              4289.953125,   4488.15478515625,    4687.5810546875,
         4888.06982421875,   5089.47900390625,   5291.68310546875,
          5494.5751953125,   5698.06103515625,   5902.05810546875]

labelcols = [str(x) for x in labelcols]

def get_nemo_mask(file_path):
    dataset = xr.open_dataset(file_path)['o2'].to_dataframe()
    dataset = dataset.unstack(level=1)
    dataset.columns = dataset.columns.levels[1]
    dataset = dataset.reset_index()
    dataset['lon_round'] = dataset['longitude']
    dataset['lat_round'] = dataset['latitude']
    dataset = dataset.drop(columns=['latitude', 'longitude', 'time'])
    columns = [str(x) for x in dataset.columns]
    dataset.columns = columns
    return dataset

def load_json(path: ty.Union[Path, str]) -> ty.Any:
    return json.loads(Path(path).read_text())


def dump_json(x: ty.Any, path: ty.Union[Path, str], *args, **kwargs) -> None:
    Path(path).write_text(json.dumps(x, *args, **kwargs) + '\n')


def load_toml(path: ty.Union[Path, str]) -> ty.Any:
    return toml.loads(Path(path).read_text())


def dump_toml(x: ty.Any, path: ty.Union[Path, str]) -> None:
    Path(path).write_text(toml.dumps(x) + '\n')


def load_pickle(path: ty.Union[Path, str]) -> ty.Any:
    return pickle.loads(Path(path).read_bytes())


def dump_pickle(x: ty.Any, path: ty.Union[Path, str]) -> None:
    Path(path).write_bytes(pickle.dumps(x))


def load(path: ty.Union[Path, str]) -> ty.Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](path)


def load_config(
    argv: ty.Optional[ty.List[str]] = None,
) -> ty.Tuple[ty.Dict[str, ty.Any], Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('-o', '--output', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--continue', action='store_true', dest='continue_')
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert args.continue_

    config_path = Path(args.config).absolute()
    output_dir = (
        Path(args.output)
        if args.output
        else config_path.parent.joinpath(config_path.stem)
    ).absolute()
    sep = '=' * (8 + max(len(str(config_path)), len(str(output_dir))))  # type: ignore[code]
    print(sep, f'Config: {config_path}', f'Output: {output_dir}', sep, sep='\n')

    assert config_path.exists()
    config = load_toml(config_path)

    if output_dir.exists():
        if args.force:
            print('Removing the existing output and creating a new one...')
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        elif not args.continue_:
            backup_output(output_dir)
            print('Already done!\n')
            sys.exit()
        elif output_dir.joinpath('DONE').exists():
            backup_output(output_dir)
            print('Already DONE!\n')
            sys.exit()
        else:
            print('Continuing with the existing output...')
    else:
        print('Creating the output...')
        output_dir.mkdir()

    environment: ty.Dict[str, ty.Any] = {}
    if torch.cuda.is_available():  # type: ignore[code]
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        pynvml.nvmlInit()
        environment['devices'] = {
            'CUDA_VISIBLE_DEVICES': cvd,
            'torch.version.cuda': torch.version.cuda,
            'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
            'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
            'driver': str(pynvml.nvmlSystemGetDriverVersion(), 'utf-8'),
        }
        if cvd:
            for i in map(int, cvd.split(',')):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                environment['devices'][i] = {
                    'name': str(pynvml.nvmlDeviceGetName(handle), 'utf-8'),
                    'total_memory': pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                }

    dump_stats({'config': config, 'environment': environment}, output_dir)
    return config, output_dir

def dump_stats(stats: dict, output_dir: Path, final: bool = False) -> None:
    dump_json(stats, output_dir / 'stats.json', indent=4)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if final:
        output_dir.joinpath('DONE').touch()
        if json_output_path:
            try:
                key = str(output_dir.relative_to(PROJECT_DIR))
            except ValueError:
                pass
            else:
                json_output_path = Path(json_output_path)
                try:
                    json_data = json.loads(json_output_path.read_text())
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    json_data = {}
                json_data[key] = stats
                json_output_path.write_text(json.dumps(json_data))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
            )


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        pass
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def set_position_sin_cos(df):
    df['lon_round_rad'] = df['lon_round'].apply(np.deg2rad)
    df['lat_round_rad'] = df['lat_round'].apply(np.deg2rad)
    df['lon_round_sin'] = df['lon_round_rad'].apply(np.sin)
    df['lon_round_cos'] = df['lon_round_rad'].apply(np.cos)
    df['lat_round_sin'] = df['lat_round_rad'].apply(np.sin)
    df['lat_round_cos'] = df['lat_round_rad'].apply(np.cos)
    return df


def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def merge_defaults(kwargs: dict, default_kwargs: dict) -> dict:
    x = deepcopy(default_kwargs)
    x.update(kwargs)
    return x


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def format_seconds(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


def get_path(relative_path: str) -> Path:
    return (
        Path(relative_path)
        if relative_path.startswith('/')
        else PROJECT_DIR / relative_path
    )

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).float() for k, v in data.items()}

def get_date_iter(start_date, end_date):
    """_summary_

    Args:
        start_date (_type_): str YYYYmmdd
        end_date (_type_): str YYYYmmdd
    """
    dt = datetime.datetime.strptime(start_date, '%Y%m%d')
    date = start_date[:]
    yield date
    while date < end_date:
        dt = dt + datetime.timedelta(days=1)
        date = dt.strftime('%Y%m%d')
        yield date

def get_month_iter(start_month, end_month):
    """_summary_

    Args:
        start_month (_type_): str YYYYmm
        end_month (_type_): str YYYYmm
    """
    dt = datetime.datetime.strptime(start_month, '%Y%m')
    month = start_month[:]
    yield month
    while month < end_month:
        dt = dt + datetime.timedelta(days=calendar.monthrange(dt.year, dt.month)[1])
        month = dt.strftime('%Y%m')
        yield month