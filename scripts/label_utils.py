import datetime
import os, glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from collections import defaultdict

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

def get_ORAS5(time=None, path=None):
    file_list = ['iicethic', 'ileadfra', 'sohefldo', 'sohtc300', 'sohtc700', 
                'sohtcbtm', 'sometauy', 'somxl010', 'somxl030', 'sosaline', 'sowaflup', 'sozotaux']

    dataframe = pd.DataFrame()
    for file_name in file_list:
        temp = xr.open_dataset(glob.glob(os.path.join(path, '*'+file_name+'*'+time+'*'))[0])
        temp = temp.to_dataframe().dropna().reset_index().drop(columns=['time_counter'])#.set_index(['lat', 'lon'])#.drop(columns=['time_counter'])#
        temp = temp.groupby(['lat', 'lon']).mean()[[file_name]]

        if len(dataframe) == 0:
            dataframe = temp
        else:
            dataframe = pd.merge(temp, dataframe, on=['lat', 'lon'])
    dataframe = dataframe.reset_index()
    dataframe = dataframe.rename(columns={'lon': 'longitude', 'lat':'latitude'})
    return dataframe
    

def get_ERA5(time=None, path=None):
    file_dict = {
        'u10': [
            'new_coord_adaptor.mars.internal-1658723200.8387213-10978-13-aa20686c-2b06-4f5e-a49f-a95462f5e4a1.nc',
            'new_coord_adaptor.mars.internal-1658722689.2003272-15839-5-a830ac6e-7b17-4921-9cb3-4aac0c99324c.nc'
        ],
        'v10': [
            'new_coord_adaptor.mars.internal-1658727580.0822122-31549-15-9e9be9fc-8287-4388-87f1-6c9d49e83b39.nc',
            'new_coord_adaptor.mars.internal-1658727614.060671-15551-13-736790c7-c416-42c4-8968-a5ee36dd4ede.nc'
        ],
        'sst': [
            'new_coord_adaptor.mars.internal-1658653663.3357341-5336-9-a2a5b06a-4608-4b4c-abc2-569eb5d85ced.nc',
            'new_coord_adaptor.mars.internal-1658985168.7370496-9986-6-92bb0a15-c86c-44e1-82dc-85d8fd3f716a.nc'
        ],
        'msnswrfcs': [
            'new_coord_adaptor.mars.internal-1658918824.3104856-15030-4-304eb10d-ddce-459f-a201-b39eca09b3ba.nc',
            'new_coord_adaptor.mars.internal-1658918797.5964792-28701-19-f2e62933-e921-43c2-b88b-0bcbb754b4b6.nc'
        ],
        'msl': [
            'new_coord_adaptor.mars.internal-1658812086.4163165-20434-11-05f145e1-69c2-483b-862f-ba6af92ca8a5.nc',
            'new_coord_adaptor.mars.internal-1659347076.3994718-20776-3-d6fd63c3-1ee0-474b-b6c1-2dd3187c2404.nc'
        ],
        'mslhf': [
            'new_coord_adaptor.mars.internal-1658985723.8585007-5474-10-522ef091-0910-4f13-a776-5bcf2c1016d9.nc',
            'new_coord_adaptor.mars.internal-1658888079.9487364-25336-17-2e7ff786-7980-4336-9a40-222ecbd7641e.nc'
        ],
        'tauoc': [
            'new_coord_adaptor.mars.internal-1659002656.9595103-18970-16-646851db-f8d6-4c60-be76-352e2ed836d8_new.nc',
            'new_coord_adaptor.mars.internal-1659002671.4926844-13499-4-6086301e-a337-4379-8fd5-b4f3244514bb_new.nc'
        ],
        'phioc': [
            'new_coord_adaptor.mars.internal-1658993924.3238332-29507-15-733f0b1c-34ad-478e-97fd-c7a557bc148a_new.nc',
            'new_coord_adaptor.mars.internal-1658993957.2352958-19238-2-b6cdb9aa-7371-400c-8dfe-f6607e0955b0_new.nc'
        ],
        't2m': [
            'new_coord_adaptor.mars.internal-1658971867.0221043-23999-16-2f9956a1-f8f1-4e7a-976b-70642b64da8a.nc',
            'new_coord_adaptor.mars.internal-1658971819.9621334-23817-4-094da1f0-1682-4250-8f2e-f0e651b4fe1b.nc'
        ],
        'tp': [
            'new_coord_adaptor.mars.internal-1658971929.1147206-13143-6-bbc2faca-1c69-46ba-9da4-3f5adf38afb3.nc',
            'new_coord_adaptor.mars.internal-1658971885.1749752-5181-20-228ace63-5119-4224-bf3d-b018786b5598.nc'
        ],
        'wmb': [
            'new_coord_adaptor.mars.internal-1659751105.3203938-26153-17-452321c2-1418-4350-a5a9-26aabda9b590_new.nc',
            'new_coord_adaptor.mars.internal-1659751076.6267784-27008-15-e8ab8131-4878-4db8-b4a1-2ae11c454484_new.nc'
        ]
    }
    dataframe = pd.DataFrame()
    for (key, values_list) in file_dict.items():
        for file_name in values_list:
            temp = xr.open_dataset(os.path.join(path, file_name))     
            hour = pd.to_datetime(temp.time.data[0]).hour
            time = pd.DatetimeIndex([datetime.datetime(time.year, time.month, time.day, hour)])[0]
            temp = temp.sel(time=time).to_dataframe().dropna().reset_index().drop(columns=['time'])
            temp = temp.rename(columns={key: key+'_'+str(hour)})
            if key in ['tauoc', 'phioc', 'wmb']:
                temp = temp.rename(columns={'lon': 'longitude', 'lat':'latitude'})
            if len(dataframe) == 0:
                dataframe = temp
            else:
                dataframe = pd.merge(temp, dataframe, on=['latitude', 'longitude'])
    return dataframe


def get_chl(time, chl_path):
    fix_name = 'ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{}-fv5.0.1.nc'
    time = time.strftime('%Y%m%d')
    name = fix_name.format(time)
    chla = xr.open_dataset(os.path.join(chl_path, name)).chlor_a.to_dataframe().reset_index().dropna()
    chla['longitude'] = chla['lon'].apply(round4)
    chla['latitude'] = chla['lat'].apply(round4)
    chla = chla.groupby(['longitude', 'latitude']).mean().reset_index().drop(columns=['lat', 'lon'])
    return chla


def round4(value):
    float_number = value - int(value)
    if 0.125<=abs(float_number)<0.375: temp = 0.25
    elif 0.375<=abs(float_number)<0.625: temp = 0.5
    elif 0.625<=abs(float_number)<0.875: temp = 0.75
    elif abs(float_number) < 0.125: temp = 0
    else:
        if value > 0: value = value + 1
        else: value = value - 1
        temp = 0
    if float_number <=0: temp = 0 - temp
    return temp + int(value)


def preprocess(data):
    columns = data.columns
    change_list = defaultdict(list)
    for col in columns:
        if '_' in col:
            name = col.split('_')[0]
            change_list[name].append(col)
    for key, value in change_list.items():
        mean = data[value].mean(axis=1)
        data[key] = mean
        data = data.drop(columns=value)
    return data


def train_validate_test_split(df, train_percent=.63, validate_percent=.07, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--save-dir', default='dataset/data', type=str,
                        help='directory to output the result')
    parser.add_argument('--oras5-path', default=None, type=str,
                        help='path to ORAS5 (default: none)')
    parser.add_argument('--era5-path', default=None, type=str,
                        help='path to ERA5 (default: none)')
    parser.add_argument('--esaoccci-path', default=None, type=str,
                        help='path to ESA OC-CCI (default: none)')
    parser.add_argument("--merge", action="store_true",
                        help="merge the driven factors with the observational DO data")
    parser.add_argument("--split", action="store_true",
                        help="split the dataset into training/validation/test set")
    args = parser.parse_args()

    if args.merge:
        labelset = pd.read_csv(os.path.join(args.save_dir, 'total_interpolate.csv'))
        labelset = labelset.rename(columns={'lat_round':'latitude', 'lon_round':'longitude'})
        dataset = pd.DataFrame()
        grouped = labelset.groupby(['time'])

        for index, (time, group) in enumerate(grouped):
            time = datetime.datetime.strptime(time, '%Y-%m-%d')
            month_time = time.strftime('%Y%m')
            oras5 = get_ORAS5(month_time, path=args.oras5_path)
            era5 = get_ERA5(time, path=args.era5_python)
            drivers = pd.merge(oras5, era5, on=['latitude', 'longitude'])
            chl = get_chl(time=time, chl_path=args.esaoccci_path)
            if chl is None:
                time += datetime.timedelta(days=1)
                continue
            drivers = pd.merge(drivers, chl, on=['latitude', 'longitude'])
            drivers = preprocess(drivers)
            frame = pd.merge(group, drivers, on=['longitude', 'latitude'])
            if len(frame) > 0:
                dataset = pd.concat([frame, dataset])
            print(index, len(dataset))
        
        dataset.to_csv(os.path.join(args.save_dir, 'dataset.csv'))
    
    if args.split:
        dataset = pd.read_csv(os.path.join(args.save_dir, 'dataset.csv'))
        train, val, test = train_validate_test_split(dataset)
        print('Train:', len(train) / len(dataset) * 100)
        print('Val:', len(val) / len(dataset) * 100)
        print('Test:', len(test) / len(dataset) * 100)
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        train.to_csv(os.path.join(args.save_dir, 'train.csv'))
        test.to_csv(os.path.join(args.save_dir, 'test.csv'))
        val.to_csv(os.path.join(args.save_dir, 'val.csv'))



