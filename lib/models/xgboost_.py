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
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from evaluate import usecols, labelcols
from lib.utils import get_path, dump_stats
from lib.dataset import Dataset, get_norm_func


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

def ln_cosh_loss(preds, dtrain):
    label =  dtrain.get_label()
    d = preds - label
    grad = np.tanh(d)/label  
    hess = (1.0 - grad*grad)/label  
    return grad, hess

def fair_obj( preds,dtrain):
    labels = dtrain.get_label() 
    con = 2
    residual = preds-labels
    grad = con*residual / (abs(residual)+con)
    hess = con**2 / (abs(residual)+con)**2
    return grad,hess


if __name__ == '__main__':

    args = {
        'seed': 1024,
        'data':{'path': 'dataset/data/Origin/', 'normalization': 'quantile', 'y_policy': 'max_min'},
        'model': {'booster' : 'gbtree', 
                'max_depth': 6,
                'min_child_weight': 8,
                'eta': 0.005,
                'subsample': 0.4,
                'colsample_bytree': 0.7,
                'lambda':0,
                # Other parameters
                'gpu_id': 0,
                'max_bin':16,
                'n_estimators':1000,
                'tree_method':'gpu_hist'},
        'fit': {'verbose': True}
    }

    zero.set_randomness(args['seed'])
    dataset_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/',args['data']['path'])
    output_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/', 'experiments/xgboost')
    
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


    norm_func = get_norm_func(dataset_dir, file_name='time_train.csv', args=args)

    test = pd.read_csv(dataset_dir / 'time_test.csv')
    feature = test[usecols].values
    feature = norm_func.transform(feature)

    train_label[np.isnan(train_label)] = -1
    val_label[np.isnan(val_label)] = -1
    
    model = XGBRegressor(objective=Pseudo_Huber_loss, **args['model'], seed=args['seed'])
  
    fit_kwargs = deepcopy(args["fit"])
    fit_kwargs['eval_set'] = [(val, val_label)]
    
    model.fit(train, train_label, **fit_kwargs)
    pred = model.predict(feature)

    pred = pred * (y_info['max'] - y_info['min']) + y_info['min']

    
    labels = test[labelcols].values
    
    mask = ~np.isnan(labels)

    truth = labels[mask]
    predictions = pred[mask]

    r2 = r2_score(truth, predictions)
    rmse = mean_squared_error(truth, predictions) ** 0.5
    mae = mean_absolute_error(truth, predictions)

    print('R2', r2, 'RMSE', rmse, 'MAE', mae)
    stats['test'] = {'R2':r2, 'RMSE':rmse, 'MAE':mae}

    # pickle.dump(model, open(output_dir/'multioutput_xgb.model', 'wb')) 
    model.save_model(output_dir / 'multioutput_xgb.json')   
    dump_stats(stats, output_dir, final=True)