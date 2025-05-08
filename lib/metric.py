import typing as ty
import numpy as np
import scipy.special
import sklearn.metrics as skm



def calculate_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    y_info: ty.Optional[ty.Dict[str, ty.Any]],
) -> ty.Dict[str, float]:
    
    mask = np.isnan(y)
    rmse = skm.mean_squared_error(y[~mask], prediction[~mask]) ** 0.5  # type: ignore[code]
    # if y_info:
    #     if y_info['policy'] == 'mean_std':
    #         rmse *= y_info['std']
    #     else:
    #         assert False
    return {'rmse': rmse, 'score': -rmse}
    


def make_summary(metrics: ty.Dict[str, ty.Any]) -> str:
    precision = 3
    summary = {}
    for k, v in metrics.items():
        if k.isdigit():
            continue
        k = {
            'score': 'SCORE',
            'accuracy': 'acc',
            'roc_auc': 'roc_auc',
            'macro avg': 'm',
            'weighted avg': 'w',
        }.get(k, k)
        if isinstance(v, float):
            v = round(v, precision)
            summary[k] = v
        else:
            v = {
                {'precision': 'p', 'recall': 'r', 'f1-score': 'f1', 'support': 's'}.get(
                    x, x
                ): round(v[x], precision)
                for x in v
            }
            for item in v.items():
                summary[k + item[0]] = item[1]

    s = [f'score = {summary.pop("SCORE"):.3f}']
    for k, v in summary.items():
        if k not in ['mp', 'mr', 'wp', 'wr']:  # just to save screen space
            s.append(f'{k} = {v}')
    return ' | '.join(s)
