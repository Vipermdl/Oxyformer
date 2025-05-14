import xarray as xr
import numpy as np
import os

def cal_lev_bnds(depth):
    depth_intervel = []
    former = 0
    for index, de in enumerate(depth):
        if index == len(depth)-1:
            latter = de
        else:
            latter = (depth[index+1] - de) / 2 + de        
        depth_intervel.append([former, latter])
        former = latter
    return np.array(depth_intervel).reshape(-1, 2)



data = xr.open_dataset('../experiments/2003-2020.nc')
mask = xr.open_dataset('../dataset/GEBCO_2022_sub_ice_topo_1440x721.nc')

elevation = 0 - mask.elevation
depth_mask = np.zeros(data.o2.shape[1:])
depth_mask = np.full_like(depth_mask, data.depth.data.reshape(-1, 1, 1))
data['o2'] = data['o2'].where(depth_mask <= elevation.data, np.nan)

temp = cal_lev_bnds(data.depth.data.tolist())
data['lev_bnds'] = xr.DataArray(temp, dims=['depth', 'bnds'], coords={'depth': data.depth.data})
data.to_netcdf('cdo_temp.nc')


os.system('cdo griddes cdo_temp.nc > mygrid')
os.system('sed -i "s/generic/lonlat/g" mygrid')
os.system('cdo setgrid,mygrid cdo_temp.nc ../experiments/mask_2003-2020.nc')
os.system('rm -rf cdo_temp.nc mygrid')

def generate_2003_2020_yearly_chlmask():
    data = xr.open_dataset('../experiments/mask_2003-2020_yearly.nc')
    mask = np.load('../mask_2003-2020_yearly_mask.npy')
    o2 = data.o2.values
    o2[:, :, ~mask] = np.nan
    data['o2'].values = o2
    data.to_netcdf('./cdo_temp.nc')
    os.system('cdo griddes ./cdo_temp.nc > mygrid')
    os.system('sed -i "s/generic/lonlat/g" mygrid')
    os.system('cdo setgrid,mygrid cdo_temp.nc mask_do_content_2003-2020_chlmask_yearly.nc')
    os.system('rm -rf mygrid cdo_temp.nc')

def generate_last_annual_proc():
    data = xr.open_dataset('mask_do_content_2003-2020_chlmask_yearly.nc')
    data['o2'].attrs['standard_name'] = 'Dissolved Oxygen'
    data['o2'].attrs['long_name'] = 'annual dissolved oxygen derived by Oxyformer'
    data['o2'].attrs['units'] = 'umol/kg'
    data['o2'].attrs['institution'] = 'East China Normal University'

    data.to_netcdf('cdo_temp.nc')
    os.system('cdo griddes ./cdo_temp.nc > mygrid')
    os.system('sed -i "s/generic/lonlat/g" mygrid')
    os.system('cdo -z zip9 setgrid,mygrid cdo_temp.nc annual_product_2003-2020_by_Oxyformer.nc')
    os.system('rm -rf mygrid cdo_temp.nc')

if __name__ == '__main__':
    generate_2003_2020_yearly_chlmask()
    generate_last_annual_proc()