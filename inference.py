import torch, os, datetime
import pandas as pd
import numpy as np
from lib.utils import load_config, get_path, set_position_sin_cos
from lib.models.model_utils import get_device, tensor
from lib.models.transformer import Transformer
from lib.models.resnet import ResNet, BasicBlock
from lib.models.mlp import MLP
from lib.dataset import Dataset, get_norm_func
from pathlib import Path
import xarray as xr
import threading
from lib.utils import  usecols, labelcols




if __name__ == '__main__':
    args, output = load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)

    device = get_device()

    dataset_dir = get_path(args['data']['path'])
    D = Dataset.from_dir(dataset_dir)
    norm_func = get_norm_func(dataset_dir, file_name='time_train.csv', args=args)
    _, y_info = D.build_y(args['data'].get('y_policy'))

    model = Transformer(
        d_numerical=D.info['n_num_features'],
        d_out=D.info['n_classes'],
        **args['model'],
    ).to(device)
    
    model.load_state_dict(torch.load(args['pretrained']['path'])['model'])
    model = model.double()
    model.eval()

    for root, dirs, files in os.walk(args['inference']['path'], topdown=False):        
        file_name = ''.join(root.split('/')[-2:])+'.nc'
        for index, name in enumerate(files):
            path = os.path.join(root, name)
            dataset = pd.read_csv(path)
            feature = dataset[usecols].values
            feature = norm_func.transform(feature)
            tensors = torch.from_numpy(feature).to(device)
            result = []
            with torch.no_grad():
                tensors_list = torch.chunk(tensors, 5)
                for inputs in tensors_list:
                    pred = model(inputs).detach()
                    result.append(pred)        
                result = torch.concat(result).cpu().numpy()
            result = result * (y_info['max'] - y_info['min']) + y_info['min']
            result = pd.DataFrame(result, columns=[float(x) for x in labelcols])
            result['lon'] = dataset['lon_round']
            result['lat'] = dataset['lat_round']
            result = result.set_index(['lat', 'lon']).stack().reset_index().rename({'level_2': 'depth', 0: 'o2'}, axis=1)
            result['time'] = datetime.datetime.strptime(name.split('.')[0], r'%Y%m%d')
            etst = result.set_index(['time', 'depth', 'lat', 'lon']).to_xarray()
            etst.to_netcdf('./cdo_temp.nc')
            os.system('cdo griddes ./cdo_temp.nc > mygrid')
            os.system('sed -i "s/generic/lonlat/g" mygrid')
            os.system('cdo setgrid,mygrid cdo_temp.nc cdo_temp_new.nc')
            if not os.path.exists('./temp'): os.makedirs('./temp')
            os.system('cdo remapbil,r1440x721 cdo_temp_new.nc ./temp/{}'.format(name.replace('csv', 'nc')))
            os.system('rm -rf mygrid cdo_temp.nc cdo_temp_new.nc')
            print(name, 'Done!')
            # if index >= 3: break
        
        os.system('cdo mergetime ./temp/*.nc temp_merge.nc')

        # data = xr.open_dataset('./temp_merge.nc')
        # mask = xr.open_dataset(args['mask']['path'])
        # elevation = 0 - mask.elevation
        # depth_mask = np.zeros(data.o2.shape[1:])
        # depth_mask = np.full_like(depth_mask, data.depth.data.reshape(-1, 1, 1))
        # data['o2'] = data['o2'].where(depth_mask <= elevation.data, np.nan)

        # temp = cal_lev_bnds(data.depth.data.tolist())
        # data['lev_bnds'] = xr.DataArray(temp, dims=['depth', 'bnds'], coords={'depth': data.depth.data})
        # data.to_netcdf('cdo_temp.nc')
        # new_file_name = 'mask_'+file_name

        root_path = output / root.split('/')[-2]
        if not os.path.exists(root_path): os.makedirs(root_path)
        
        # os.system('cdo griddes cdo_temp.nc > mygrid')
        # os.system('sed -i "s/generic/lonlat/g" mygrid')
        # os.system('cdo setgrid,mygrid cdo_temp.nc cdo_temp_new.nc')#.format(os.path.join(root_path, new_file_name)))
        os.system('cdo monmean temp_merge.nc {}'.format(os.path.join(root_path, file_name)))
        os.system('rm -rf temp ./temp_merge.nc') 
