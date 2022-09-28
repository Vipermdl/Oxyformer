# Deep learning for eastimating dissolved oxygen in global ocean

## Introduction
Dissolved oxygen (DO) has far-reaching impacts on ocean biodiversity, productivity, and biogeochemical cycles, showing a decline trend by in-situ measurements and model simulation during the past five decades. Comprehensive understanding of DO is hindered by limited measurement data and inadequate spatial resolution or skill of process-based models. We built a transformer-based (Oxyformer) deep learning method using hydrometeorological and biogeochemical driver data to estimate global ocean DO concentration. Oxyformer provides daily DO concentration estimation throughout whole water column at spatial resolution of 0.25° × 0.25°. Moreover, Oxyformer’s accuracy is higher than other methods in the areas with different driven factors (such as at open ocean and coastal water), especially overcoming weakness of overestimation for DO concentration in hypoxia regions. An interpretable analysis indicates Oxyformer employed plausible drivers to estimate DO concentration. The Oxyformer model is a powerful tool for both oceanic DO concentration estimation and for modeling its complex DO variations on marine-system.

<div style="color:#0000FF" align="center">
<img src="new_model - 6.2.jpg" width="680"/>
</div>

## Major features
- **Design a transformer-based model (Oxyformer) to estimate oceanic DO concentration at global scale.**
- **Oxyformer have higher performance in terms of consistency with historical DO concentration measurements compared to physical-based model and other machine learning methods.**
- **Oxyformer's variables importance results are consistent with established causal linkages between hydrometeorological, biogeochemical drivers and DO concentration.**
- **The spatial correlation and temporal trend are trustworthy displayed by Oxyformer.**

## Benchmarking

In this work, we integrated DO observational data from different data source to a more extensive database in combination with using rigorously depth mapping and strict quality control. And DO dataset are then combined for data-driven modeling using observed DO measurements as the response variable and hydrometeorological / biogeochemical drivers as variables. Details are illustrated in our paper. 

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

