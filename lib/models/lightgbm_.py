from pathlib import Path
from random import seed
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import typing as ty

import sys, zero, pickle
from copy import deepcopy

sys.path.append('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from evaluate import usecols, labelcols
from lib.utils import get_path, dump_stats
from lib.dataset import Dataset
from main import get_norm_func


def Pseudo_Huber_loss(real, predict): 
    predict = predict.reshape(real.shape)
    
    predict[real < 0] = -1 + 1e-6
    real[real < 0] = -1 + 1e-6
    
    d = predict - real 
    h = 1 #h is delta in the formula  
    scale = 1 + (d / h) ** 2  
    scale_sqrt = np.sqrt(scale)  
    grad = d / scale_sqrt  
    hess = 1 / scale / scale_sqrt
    return grad, hess

if __name__ == '__main__':

    args = {
        'seed': 1024,
        'data':{'path': 'dataset/data/Origin/', 'normalization': 'quantile', 'y_policy': 'max_min'},
        'model': {
                'colsample_bytree':0.7520316396425557,
                'learning_rate': 0.03012327705724217,
                'min_child_samples': 8,
                'min_child_weight': 0.00794616343686661,
                'n_estimators': 500,
                'n_jobs': 1,
                'num_leaves': 40,
                'reg_lambda': 0.003343640211399975,
                'subsample': 0.9425389542354896},
        'fit':{
            'eval_metric': 'rmse',
            'verbose': True
        }
    }

    zero.set_randomness(args['seed'])
    dataset_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/',args['data']['path'])
    output_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/', 'experiments/lightgbm')
    
    if not output_dir.exists():
        output_dir.mkdir()
    
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **args,
    }
    timer = zero.Timer()
    timer.run()

    D = Dataset.from_dir(dataset_dir)

    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        seed=args['seed'],
    )
    
    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))

    train, val = X['train'], X['val']
    train_label, val_label = Y['train'], Y['val']

    norm_func = get_norm_func(dataset_dir, norm=args['data']['normalization'], seed=args['seed'])

    test = pd.read_csv(dataset_dir / 'time_test.csv')
    feature = test[usecols].values
    feature = norm_func.transform(feature)


    train_label[np.isnan(train_label)] = -1
    val_label[np.isnan(val_label)] = -1

    model_kwargs = deepcopy(args['model'])
    fit_kwargs = deepcopy(args['fit'])
    # fit_kwargs['eval_set'] = [(val, val_label)]
    
    model = LGBMRegressor(objective=Pseudo_Huber_loss, **model_kwargs)

    # import pdb; pdb.set_trace()
    multioutput = MultiOutputRegressor(model).fit(train, train_label, **fit_kwargs)

    pred = multioutput.predict(feature)

    pred = pred * (y_info['max'] - y_info['min']) + y_info['min']
    
    import pdb; pdb.set_trace()

    labels = test[labelcols].values
    
    mask = ~np.isnan(labels)

    truth = labels[mask]
    predictions = pred[mask]

    r2 = r2_score(truth, predictions)
    rmse = mean_squared_error(truth, predictions) ** 0.5
    mae = mean_absolute_error(truth, predictions)

    print('R2', r2, 'RMSE', rmse, 'MAE', mae)
    stats['test'] = {'R2':r2, 'RMSE':rmse, 'MAE':mae}

    pickle.dump(multioutput, open(output_dir/'multioutput_lightgbm.pkl', 'wb'))  
    dump_stats(stats, output_dir, final=True)

   