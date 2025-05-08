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
from sklearn.ensemble import RandomForestRegressor
from evaluate import usecols, labelcols
from lib.utils import get_path, dump_stats
from lib.dataset import Dataset
from main import get_norm_func


def myloss(pred, truth):
    import pdb; pdb.set_trace()

if __name__ == '__main__':

    args = {
        'seed': 1024,
        'data':{'path': 'dataset/data/Origin/', 'normalization': 'quantile', 'y_policy': 'max_min'},
        'model': {'n_estimators':2000,
                    # 'criterion':'squared_error',
                    'max_depth':6,
                    'max_features':0.8,
                    'min_samples_leaf':8,
                    'n_jobs':12},
    }

    zero.set_randomness(args['seed'])
    dataset_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/',args['data']['path'])
    output_dir = Path('/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/', 'experiments/randomforeset')
    
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
    
    # model = XGBRegressor(objective=Pseudo_Huber_loss, **args['model'], seed=args['seed'])
    model = RandomForestRegressor(criterion=myloss, random_state=args['seed'], **args['model'])
  
    # fit_kwargs = deepcopy(args["fit"])
    # fit_kwargs['eval_set'] = [(val, val_label)]
    
    model.fit(train, train_label)#, **fit_kwargs
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

    # pickle.dump(multioutput, open(output_dir/'multioutput_xgb.model', 'wb')) 
    model.save(output_dir/'multioutput_rf.model')   
    dump_stats(stats, output_dir, final=True)

    # predictions = []
    # for index in range(train_label.shape[1]):
    #     label = train_label[:, index]
    #     mask = ~np.isnan(label)

    #     # if np.all(~mask):
    #     #     continue
    #     print(index)

    #     train_index = train[mask, :]
    #     train_label_index = label[mask]

    #     model = XGBRegressor( **args['model'], seed=args['seed'], objective=loss)

    #     label = val_label[:, index]
    #     mask = ~np.isnan(label)
    #     val_index = val[mask, :]
    #     val_label_index = label[mask]

    #     fit_kwargs = deepcopy(args["fit"])
    #     fit_kwargs['eval_set'] = [(val_index, val_label_index)]

    #     model.fit(train_index, train_label_index, **fit_kwargs)

    #     pred = model.predict(feature)

    #     predictions.append(pred)
    
    # temp = np.vstack(predictions).T
    # pred = temp * (y_info['max'] - y_info['min']) + y_info['min']
    # labels = test[labelcols].values

    # mask = ~np.isnan(labels)
    # truth = labels[mask]
    # predictions = pred[mask]
    
    # r2 = r2_score(truth, predictions)
    # rmse = mean_squared_error(truth, predictions) ** 0.5
    # mae = mean_absolute_error(truth, predictions)

    # print('R2', r2, 'RMSE', rmse, 'MAE', mae)
    # stats['test'] = {'R2':r2, 'RMSE':rmse, 'MAE':mae}
 
    # dump_stats(stats, output_dir, final=True)

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()


    
    
    

