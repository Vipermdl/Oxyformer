"""_summary_

Returns:
    depth_x: _description_
"""
import gsw
import argparse
import os, glob
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import interpolate

variable = ["Depth     ", "Temperatur ", "Oxygen     ", 'Salinity   ']
depth_x = [ 0.5057600140571594, 1.5558550357818604,  2.667681932449341, 3.8562800884246826, 5.1403608322143555,  6.543034076690674,
            8.09251880645752,  9.822750091552734, 11.773679733276367, 13.991040229797363, 16.525320053100586, 19.429800033569336, 22.757619857788086, 
            26.558300018310547,  30.87455940246582, 35.74020004272461,  41.18001937866211, 47.211891174316406, 53.85063934326172,  61.11283874511719,  
            69.02168273925781, 77.61116027832031,  86.92942810058594,  97.04131317138672, 108.0302963256836,              120.0,  133.0758056640625,
            147.4062042236328,  163.1645050048828, 180.54989624023438, 199.7899932861328, 221.14120483398438, 244.89059448242188, 271.3564147949219, 
            300.88751220703125,    333.86279296875, 370.6885070800781,  411.7939147949219,  457.6256103515625, 508.639892578125,  565.2922973632812,  
            628.0260009765625, 697.2587280273438,  773.3682861328125,  856.6790161132812, 947.4478759765625,   1045.85400390625,  1151.990966796875,
            1265.8609619140625,     1387.376953125,  1516.364013671875, 1652.5679931640625, 1795.6710205078125, 1945.2960205078125, 2101.027099609375,  
            2262.422119140625,   2429.02490234375, 2600.3798828125,       2776.0390625,  2955.570068359375, 3138.56494140625,   3324.64111328125,  
            3513.446044921875, 3704.656982421875,   3897.98193359375,  4093.158935546875, 4289.953125,   4488.15478515625,    4687.5810546875,
            4888.06982421875,   5089.47900390625,   5291.68310546875, 5494.5751953125,   5698.06103515625,   5902.05810546875]


def searchInsert(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right-left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def convert_df(cast_list,  year, month, day, time, longitude, latitude):
    variable_name = cast_list[0].split(",")
    dict_variable = defaultdict(list)
    for index, line in enumerate(cast_list[3:]):
        variable_content = line.split(",")
        for var, name in zip(variable_content, variable_name):
            if name in variable:
                dict_variable[name.strip()].append(var.strip()) 
    
    cast_df = pd.DataFrame.from_dict(dict_variable)
    cast_df[cast_df=='---*---'] = np.nan
    cast_df[cast_df=='**********'] = np.nan
    cast_df = cast_df.dropna()
    cast_df['Depth'] = cast_df['Depth'].astype(np.float64)
    cast_df['Temperatur'] = cast_df['Temperatur'].astype(np.float64)
    cast_df['Oxygen'] = cast_df['Oxygen'].astype(np.float64)
    cast_df['Salinity'] = cast_df['Salinity'].astype(np.float64)
    
    cast_df = cast_df.groupby(['Depth']).mean().reset_index()
    
    if cast_df['Oxygen'].min() < 0 or cast_df['Oxygen'].max() > 600:
        return None
    
    Oxygen = cast_df['Oxygen'].values
    Depth = cast_df['Depth'].values
    Temperature = cast_df['Temperatur'].values
    Salinity = cast_df['Salinity'].values
    depth_list = cast_df['Depth'].tolist()

    sorted_id = sorted(range(len(depth_list)), key=lambda k: depth_list[k], reverse=False)   
    Depth = Depth[sorted_id]
    Oxygen = Oxygen[sorted_id]
    Temperature = Temperature[sorted_id]
    Salinity = Salinity[sorted_id]
    
    if len(Oxygen)<=2 or len(Salinity)<=2 or len(Temperature)<=2: return None
    
    end_index = searchInsert(depth_x, Depth[-1])
    start_index = searchInsert(depth_x, Depth[0])
    
    oxy_func = interpolate.PchipInterpolator(Depth, Oxygen)
    oxygen_inter = oxy_func(np.array(depth_x[start_index:end_index]))
    
    tem_func = interpolate.PchipInterpolator(Depth, Temperature)
    tem_inter = tem_func(np.array(depth_x[start_index:end_index]))
    
    sal_func = interpolate.PchipInterpolator(Depth, Salinity)
    sal_inter = sal_func(np.array(depth_x[start_index:end_index]))
    
    if np.isnan(oxygen_inter).all() or oxygen_inter.max() > 600 or oxygen_inter.min() < 0:
        return None
    
    #Additional quality control measures were the removal of profiles with supersaturation at depths deeper than 200 m as well as supersaturation above 115%.
    saturation = gsw.O2sol_SP_pt(sal_inter, tem_inter)
    depth_gt_200 = np.array(depth_x[start_index:end_index]) > 200
    depth_gt_200_flag = [o < s for o, s in zip(oxygen_inter[depth_gt_200], saturation[depth_gt_200])]
    o2sat_115 = saturation * 1.15
    o2sat_115_flag = [o < s for o, s in zip(oxygen_inter, o2sat_115)]
    if not (np.all(depth_gt_200_flag) and np.all(o2sat_115_flag)):
        return None
    
    temp_oxygen = [np.nan for x in depth_x]
    temp_oxygen[start_index:end_index] = oxygen_inter
    info = dict(zip(depth_x, temp_oxygen))
    
    temp_sal = [np.nan for x in depth_x]
    temp_sal[start_index:end_index] = sal_inter
    depth_sal = ['sal_'+str(x) for x in depth_x]
    info.update(dict(zip(depth_sal, temp_sal)))
    
    temp_ter = [np.nan for x in depth_x]
    temp_ter[start_index:end_index] = tem_inter
    depth_tem = ['tem_'+str(x) for x in depth_x]
    info.update(dict(zip(depth_tem, temp_ter)))
    
    if longitude == '' or latitude == '':
        return None
    
    info['year'] = year
    info['month'] = month
    info['day'] = day
    info['time'] = time
    info['longitude'] = float(longitude)
    info['latitude'] = float(latitude)

    info = pd.Series(info).to_frame().T
    return info


def func(cast_list):
    year, month, day, time, longitude, latitude =np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    for index,line in enumerate(cast_list):
        if "Latitude" in line:
            latitude = line.split(",,")[1].strip().replace(",decimal degrees", "")
        if "Longitude" in line:
            longitude = line.split(",,")[1].strip().replace(",decimal degrees", "")
        if "Year" in line:
            year = line.split(",,")[1].strip()
        if "Month" in line:
            month = line.split(",,")[1].strip()
        if "Day" in line:
            day = line.split(",,")[1].strip()
        if "Time" in line:
            time = line.split(",,")[1].strip().replace(",decimal hours (UT)", "")
        if 'UNITS' in line:
            if not ('umol/kg' in line and 'degrees C' in line):
                return None
        if "VARIABLES " in line:
            name_list = line.split(",")
            if set(variable).issubset(name_list):
                return convert_df(cast_list[index:], year, month, day, time, longitude, latitude)
            else:
                return None


def interpolate_WOD(args):
    csv_list = os.listdir(args.data_wod_path)
    for csv_file_name in csv_list:
        with open(os.path.join(args.data_path, csv_file_name), "r") as reader:
            contents = reader.readlines()      
        result = pd.DataFrame()
        cast_list = []
        csv_name = csv_name.split('/')[-1]
        for con in contents:
            if "#--------------------------------------------------------------------------------" in con:
                cast_list.clear()
            elif "END OF VARIABLES SECTION," in con:
                case_df = func(cast_list)
                if case_df is not None:
                    result = pd.concat([result, case_df])
                    print(result, csv_name)
            else:
                cast_list.append(con)
        result.to_csv(os.path.join(args.save_dir, csv_name))
    
    csv_file_list = glob.glob(os.path.join(args.save_dir, "*.csv"))
    merge_DF = pd.DataFrame()
    for csv_name in csv_file_list:
        # source = csv_name.split('.')[-2][:3]
        df = pd.read_csv(csv_name)
        # df['source'] = source
        merge_DF = pd.concat([merge_DF, df])
    merge_DF = merge_DF.reset_index()
    merge_DF.to_csv(os.path.join(args.save_dir, 'WOD_interpolate.csv'))

 
def interpolate_GLODAPV2(args):
    dataset = pd.read_csv(os.path.join(args.data_glodapv2_path, 'GLODAPv2.2021_Merged_Master_File.csv')).groupby(['G2year', 'G2month',  'G2day', 'G2latitude',  'G2longitude'])
    result = pd.DataFrame()
    for index, ((year, month, day, latitude, longitude), group) in enumerate(dataset):
        group = group[['G2depth', 'G2oxygen', 'G2salinity', 'G2temperature']]
        group[group == -9999.00] = np.nan
        group = group.dropna()
        group['G2depth'] = group['G2depth'].astype(np.float64)
        group['G2oxygen'] = group['G2oxygen'].astype(np.float64)
        group['G2salinity'] = group['G2salinity'].astype(np.float64)
        group['G2temperature'] = group['G2temperature'].astype(np.float64)

        group = group.groupby(['G2depth']).mean().reset_index()
        if group['G2oxygen'].min() < 0 or group['G2oxygen'].max() > 600: continue
        Oxygen = group['G2oxygen'].values
        Depth = group['G2depth'].values
        Temperature = group['G2temperature'].values
        Salinity = group['G2salinity'].values
        depth_list = group['G2depth'].tolist()

        time = datetime.datetime(year==year, month=month, day=day)
        sorted_id = sorted(range(len(depth_list)), key=lambda k: depth_list[k], reverse=False)   
        Depth = Depth[sorted_id]
        Oxygen = Oxygen[sorted_id]
        Temperature = Temperature[sorted_id]
        Salinity = Salinity[sorted_id]
        
        info = convert(Depth, Oxygen, Temperature, Salinity, longitude, latitude, time)
        
        if info is None: continue
        else: result = pd.concat([result, info])
        print(result, index)
    result.to_csv(os.path.join(args.save_dir, 'GLODAPV2_interpolate.csv'))


def convert(Depth, Oxygen, Temperature, Salinity, longitude, latitude, time):
    if len(Oxygen)<=2 or len(Salinity)<=2 or len(Temperature)<=2: return None
    end_index = searchInsert(depth_x, Depth[-1])
    start_index = searchInsert(depth_x, Depth[0])
    oxy_func = interpolate.PchipInterpolator(Depth, Oxygen)
    oxygen_inter = oxy_func(np.array(depth_x[start_index:end_index]))
    tem_func = interpolate.PchipInterpolator(Depth, Temperature)
    tem_inter = tem_func(np.array(depth_x[start_index:end_index]))
    sal_func = interpolate.PchipInterpolator(Depth, Salinity)
    sal_inter = sal_func(np.array(depth_x[start_index:end_index]))

    if np.isnan(oxygen_inter).all() or oxygen_inter.max() > 600 or oxygen_inter.min() < 0: return None
    
    #Additional quality control measures were the removal of profiles with supersaturation at depths deeper than 200 m as well as supersaturation above 115%.
    saturation = gsw.O2sol_SP_pt(sal_inter, tem_inter)
    depth_gt_200 = np.array(depth_x[start_index:end_index]) > 200
    depth_gt_200_flag = [o < s for o, s in zip(oxygen_inter[depth_gt_200], saturation[depth_gt_200])]
    o2sat_115 = saturation * 1.15
    o2sat_115_flag = [o < s for o, s in zip(oxygen_inter, o2sat_115)]
    if not (np.all(depth_gt_200_flag) and np.all(o2sat_115_flag)): return None
    
    temp_oxygen = [np.nan for x in depth_x]
    temp_oxygen[start_index:end_index] = oxygen_inter
    info = dict(zip(depth_x, temp_oxygen))
    
    temp_sal = [np.nan for x in depth_x]
    temp_sal[start_index:end_index] = sal_inter
    depth_sal = ['sal_'+str(x) for x in depth_x]
    info.update(dict(zip(depth_sal, temp_sal)))
    
    temp_ter = [np.nan for x in depth_x]
    temp_ter[start_index:end_index] = tem_inter
    depth_tem = ['tem_'+str(x) for x in depth_x]
    info.update(dict(zip(depth_tem, temp_ter)))
    
    if longitude == '' or latitude == '': return None
    
    info['year'] = time.year
    info['month'] = time.month
    info['day'] = time.day
    info['longitude'] = float(longitude)
    info['latitude'] = float(latitude)
    
    info = pd.Series(info).to_frame().T
    return info


def interpolate_Pangaea(args):
    dataset = pd.read_csv(os.path.join(args.data_pangaea_path, 'pangaea_merge.csv')).dropna().groupby(['Time', 'Latitude', 'Longitude'])
    result = pd.DataFrame()

    for index, ((time, latitude, longitude), group) in enumerate(dataset):
        group = group[['Depth', 'O2', 'Sal', 'Temp']]
        group['Depth'] = group['Depth'].astype(np.float64)
        group['O2'] = group['O2'].astype(np.float64)
        group['Sal'] = group['Sal'].astype(np.float64)
        group['Temp'] = group['Temp'].astype(np.float64)

        group = group.groupby(['Depth']).mean().reset_index()

        if group['O2'].min() < 0 or group['O2'].max() > 600: continue

        Oxygen = group['O2'].values
        Depth = group['Depth'].values
        Temperature = group['Temp'].values
        Salinity = group['Sal'].values

        # Oxygen convert μmol/L (micromoles per liter) to μmol/kg
        Oxygen = Oxygen * (1000 / 1027)
        # import pdb; pdb.set_trace()
        time = datetime.datetime.strptime(time, '%Y-%m-%d')
        depth_list = group['Depth'].tolist()
        sorted_id = sorted(range(len(depth_list)), key=lambda k: depth_list[k], reverse=False)   
        Depth = Depth[sorted_id]
        Oxygen = Oxygen[sorted_id]
        Temperature = Temperature[sorted_id]
        Salinity = Salinity[sorted_id]
        
        info = convert(Depth, Oxygen, Temperature, Salinity, longitude, latitude, time)
        
        if info is None: continue
        else: result = pd.concat([result, info])
        print(result, index)
    result.to_csv(os.path.join(args.save_dir, 'PANGAEA_interpolate.csv'))


def interpolate_cchd(args):
    dataset = pd.read_csv(os.path.join(args.data_cchd_path, 'cchd.csv'))
    dataset['DATE'] = dataset['DATE'].astype(np.int64).astype(np.str_)
    dataset['DATE'] = pd.to_datetime(dataset['DATE'], errors='coerce').dropna()

    dataset = dataset.groupby(['DATE', 'LATITUDE', 'LONGITUDE'])
    result = pd.DataFrame()

    for index, ((time, latitude, longitude), group) in enumerate(dataset):
        group = group[['DEPTH', 'OXYGEN', 'CTDSAL', 'CTDTMP']]
        group['DEPTH'] = group['DEPTH'].astype(np.float64)
        group['OXYGEN'] = group['OXYGEN'].astype(np.float64)
        group['CTDSAL'] = group['CTDSAL'].astype(np.float64)
        group['CTDTMP'] = group['CTDTMP'].astype(np.float64)
        group = group.groupby(['DEPTH']).mean().reset_index()
        if group['OXYGEN'].min() < 0 or group['OXYGEN'].max() > 600: continue
        Oxygen = group['OXYGEN'].values
        Depth = group['DEPTH'].values
        Temperature = group['CTDTMP'].values
        Salinity = group['CTDSAL'].values

        time = datetime.datetime.strptime(str(time), '%Y%m%d')
        
        depth_list = group['DEPTH'].tolist()
        sorted_id = sorted(range(len(depth_list)), key=lambda k: depth_list[k], reverse=False)   
        Depth = Depth[sorted_id]
        Oxygen = Oxygen[sorted_id]
        Temperature = Temperature[sorted_id]
        Salinity = Salinity[sorted_id]
        
        info = convert(Depth, Oxygen, Temperature, Salinity, longitude, latitude, time)
        
        if info is None: continue
        else: result = pd.concat([result, info])
        result = pd.concat([result, info])
        print(result, index)
    result.to_csv(os.path.join(args.save_dir, 'CCHD_interpolate.csv'))

    
def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--save-dir', default='interpolate_result', type=str,
                        help='directory to output the result')
    parser.add_argument('--data-wod-path', default=None, type=str,
                        help='path to World Ocean Database (default: none)')
    parser.add_argument('--data-glodapv2-path', default=None, type=str,
                        help='path to Global Ocean Data Analysis (default: none)')
    parser.add_argument('--data-pangaea-path', default=None, type=str,
                        help='path to Pangaea Database (default: none)')
    parser.add_argument('--data-cchd-path', default=None, type=str,
                        help='path to CLIVAR and Carbon Hydrographic Database (default: none)')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    if not args.data_wod_path is None:
        interpolate_WOD(args=args)
    if not args.data_glodapv2_path is None:
        interpolate_GLODAPV2(args=args)
    if not args.data_pangaea_path is None:
        interpolate_Pangaea(args=args)
    if not args.data_cchd_path is None:
        interpolate_cchd(args=args)
    
    csv_file_list = glob.glob(os.path.join(args.save_dir, "*_interpolate.csv"))
    merge_DF = pd.DataFrame()
    for csv_name in csv_file_list:
        df = pd.read_csv(csv_name)
        merge_DF = pd.concat([merge_DF, df])
    merge_DF = merge_DF.reset_index()
    merge_DF.to_csv(os.path.join(args.save_dir, 'total_interpolate.csv'))


if __name__ == "__main__":
    main()

    

