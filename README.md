# Deep learning for eastimating dissolved oxygen in global ocean

## Introduction




### Results and Models
|  Method   | $R^{2}$ | MAE | RMSE |
| :---------: | :-----: | :------: | :------------: | 
|    NEMO-PICES     |   0.78    |   21.481    |      37.428      | 
|    Light-GBM    |   0.83    |   20.291    |      32.682      |  
| XGBoost |   0.8    |   22.302    |      35.245      | 
| MLP |   0.78    |   24.324    |      37.021      | 
| ResNet |   0.84    |   20.073    |      31.379      | 
| Oxyformer (ours) |   0.86    |   18.581   |      29.835       |  


## Preparation

First of all, clone the code
```
git clone https://github.com/Vipermdl/Oxyformer
```

### prerequisites

* Python 3.8
* Pytorch 1.10.1
* CUDA 11.3 or higher

### Data Preparation

The dataset will upload if it is available.

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

### Train

Try:
```
python main.py experiments/Oxyformer.toml
```

### Test

If you want to evlauate the detection performance, simply run
```
python evaluate.py experiments/inference.toml
```

### Inference

```
python inference.py experiments/inference.toml -o outputdir
```

