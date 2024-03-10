import os
import argparse
import numpy as np
import pandas as pd



depth = [ '0.5057600140571594', '1.5558550357818604', '2.667681932449341', '3.8562800884246826', '5.1403608322143555', '6.543034076690674', '8.09251880645752', '9.822750091552734',
       '11.773679733276367', '13.991040229797363', '16.525320053100586', '19.429800033569336', '22.757619857788086', '26.558300018310547', '30.87455940246582', '35.74020004272461', '41.18001937866211',
       '47.211891174316406', '53.85063934326172', '61.11283874511719', '69.02168273925781', '77.61116027832031', '86.92942810058594', '97.04131317138672', '108.0302963256836', '120.0', '133.0758056640625',
       '147.4062042236328', '163.1645050048828', '180.54989624023438', '199.7899932861328', '221.14120483398438', '244.89059448242188', '271.3564147949219', '300.88751220703125', '333.86279296875',
       '370.6885070800781', '411.7939147949219', '457.6256103515625', '508.639892578125', '565.2922973632812', '628.0260009765625', '697.2587280273438', '773.3682861328125', '856.6790161132812',
       '947.4478759765625', '1045.85400390625', '1151.990966796875', '1265.8609619140625', '1387.376953125', '1516.364013671875', '1652.5679931640625', '1795.6710205078125', '1945.2960205078125',
       '2101.027099609375', '2262.422119140625', '2429.02490234375', '2600.3798828125', '2776.0390625', '2955.570068359375', '3138.56494140625', '3324.64111328125', '3513.446044921875',
       '3704.656982421875', '3897.98193359375', '4093.158935546875', '4289.953125', '4488.15478515625', '4687.5810546875', '4888.06982421875', '5089.47900390625', '5291.68310546875',
       '5494.5751953125', '5698.06103515625', '5902.05810546875']


def merge_subgroup(data):
    info = {}
    for name in depth:
        value = data[str(name)]
        mask =  value.isnull()
        value = value[~mask]
        if mask.all():
            info[str(name)] = np.nan
        else:
            if value.dtype == np.str_: return None
            info[str(name)] = value.astype(np.float64).mean()
    if 'source' in data.columns.tolist(): info['source'] = '-'.join(data['source'].unique().tolist())
    if 'year' in data.columns.tolist(): info['year'] = data['year'].mean()
    if 'month' in data.columns.tolist(): info['month'] = data['month'].mean()
    if 'day' in data.columns.tolist(): info['day'] = data['day'].mean()
    if 'longitude' in data.columns.tolist(): info['longitude'] = data['longitude'].mean()
    if 'latitude' in data.columns.tolist(): info['latitude'] = data['latitude'].mean()
    if 'Unnamed: 0' in data.columns.tolist(): info['Unnamed: 0'] = data['Unnamed: 0'].mean()
    return pd.Series(info).to_frame().T
        

def remove_duplicate_profile(data):
    """
    remove duplicate profiles within 5km and 25h, the profile with the best vertical resolution was used.
    """
    data['daily_res'] = data['year'].astype(np.str_)+'-'+data['month'].astype(np.str_)+'-'+data['day'].astype(np.str_)
    grouped = data.groupby(['daily_res', 'lon_5km', 'lat_5km'])
    result = pd.DataFrame()
    for id, ((daily_res, lon_5km, lat_5km), group) in enumerate(grouped):
        group = group.drop(columns=['daily_res', 'lon_5km', 'lat_5km', 'Unnamed: 0'])
        if len(group) > 1:
            flag = group[depth].isna().sum(axis=1)
            group = group[flag == flag.min()]
            if len(group) > 1:
                group = merge_subgroup(group) # if there are multiple profiles with the highest vertical resolution, we employ the mean way to get it.
                if group is None: continue
        result = pd.concat([result, group]) 
    #result.drop(columns=['Unnamed: 0']).reset_index(drop=True, inplace=True)
    import pdb; pdb.set_trace()
    result.reset_index(drop=True, inplace=True)
    return result


def remove_max_min_within_5(data):
    """
    profiles with a difference of less than 5 umol/kg between maximum and minimum observed oxygen were remove.
    """
    max = data[depth].max(axis=1)
    min = data[depth].min(axis=1)
    flag = (max - min) >= 5
    data = data[flag]
    data.reset_index(drop=True, inplace=True)
    return data


def remove_continuous_18_depth_within_05(data):
    """
    profiles with oxygen difference of less than 0.5 umol/kg within 18 depth levels was remove.
    """
    def check_continuous_18_depth(Oxygen):
        Oxygen = Oxygen[np.isnan(Oxygen.values)]
        del_flag = False
        if len(Oxygen) >= 18:
            end_index = len(Oxygen) - 18 + 1
            fx = lambda x: (x.max() - x.min()) < 0.5
            temp = [fx(Oxygen[index:index+18]) for index in range(end_index)]
            if True in temp: del_flag = True
        return del_flag
    
    del_flag = data[depth].apply(check_continuous_18_depth, axis=1)
    data = data[~del_flag]
    data.reset_index(drop=True, inplace=True)
    return data


def remove_surface_less_than_100(data):
    """
    profiles with less than 100 umol/kg oxygen at the surface were removed.
    """
    flag = data[depth[0]] >= 100
    nan_flag = data[depth[0]].isna()
    data = data[flag | nan_flag]
    data.reset_index(drop=True, inplace=True)
    return data


def remove_supersaturation_above_115_deper_than_200m(data):
    """
    profiles with supersaturation at depths deepers than 200m as well as supersaturation above 115%.
    """
    pass
    import pdb; pdb.set_trace()


def round4(value):
    "get the space profile with 0.25"
    float_number = value - int(value)
    if 0.125<=abs(float_number)<0.375:
        temp = 0.25
    elif 0.375<=abs(float_number)<0.625:
        temp = 0.5
    elif 0.625<=abs(float_number)<0.875:
        temp = 0.75
    elif abs(float_number) < 0.125:
        temp = 0
    else:
        if value > 0:
            value = value + 1
        else:
            value = value - 1
        temp = 0
    if float_number <=0:
        temp = 0 - temp
    return temp + int(value)


def searchInsert(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right-left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def round_5km_lat(lat):
    lat_array = list(np.load('lat.npy'))
    lat_array.reverse()
    if float(lat) < lat_array[0]: 
        lat = lat_array[0]
    elif float(lat) > lat_array[-1]:
        lat = lat_array[-1]
    else:
        lat = float(lat)
    return lat_array[searchInsert(lat_array, lat)]


def round_5km_lon(lon):
    lon_array = list(np.load('lon.npy'))
    if float(lon) < lon_array[0]: 
        lon = lon_array[0]
    elif float(lon) > lon_array[-1]:
        lon = lon_array[-1]
    else:
        lon = float(lon)
    return lon_array[searchInsert(lon_array, lon)]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--save-dir', default='interpolate_result', type=str,
                        help='directory to output the result')
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.save_dir, 'total_interpolate.csv'))

    data = data.drop(columns=['time'])
    data = data[data['year']>=2002]
    data['lon_5km'] = data['longitude'].apply(round_5km_lon)
    data['lat_5km'] = data['latitude'].apply(round_5km_lat)

    data = remove_duplicate_profile(data)
    print('remove duplicate profile Done!')
    data = remove_max_min_within_5(data)
    print('remove max min with 5 Done!')
    data = remove_continuous_18_depth_within_05(data)
    print('remove continuous 18 depth levels within 5 Done!')
    
    data = remove_surface_less_than_100(data)
    print('remove surface less than 100 Done!')
    
    data.to_csv(os.path.join(args.save_dir, 'temp.csv')) # to save the temp result
    
    # import pdb; pdb.set_trace()
    # data = pd.read_csv(os.path.join(args.save_dir, 'temp.csv'))

    data = remove_supersaturation_above_115_deper_than_200m(data)
    
    data[~((data[depth]>=0) & (data[depth]<=600))] = np.nan
    data = data[((data['longitude']>=-180) & (data['longitude']<=180))]
    data = data[((data['latitude']>=-90) & (data['latitude']<=90))]
    data['lon_round'] = data['longitude'].apply(round4)
    data['lat_round'] = data['latitude'].apply(round4)
    
    data['time'] = pd.to_datetime(data[['year', 'month', 'day']], errors='coerce')
    mask = data['time'].isna()
    data = data[~mask]
    data = data[(data['year']>=2002) & (data['year']<=2020)]
    
    data = data.drop(columns=['Unnamed: 0',  'Unnamed: 0.1'])

    # remap 025 degree
    grouped = data.groupby(['time', 'lon_round', 'lat_round'])
    result = pd.DataFrame()
    for index, ((time, lon_round, lat_round), group) in enumerate(grouped):
        print(index)
        if len(group) > 1:
            info = {}
            info['time'] = time
            info['lon_round'] = lon_round
            info['lat_round'] = lat_round
            for de in depth:
                value = group[str(de)]
                mask =  value.isnull()
                value = value[~mask]
                if mask.all(): info[str(de)] = np.nan
                else: info[str(de)] = value.mean()
            group = pd.Series(info).to_frame().T
        else: group = group.drop(columns=['year', 'month', 'day', 'longitude', 'latitude'])
        group[depth] = group[depth].astype(np.float64)
        result = pd.concat([result, group])
    
    result.reset_index(inplace=True, drop=True)
    result.to_csv(os.path.join(args.save_dir, 'new_interpolate_2002-2020_sqc_nemo_depth.csv'))