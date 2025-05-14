from random import sample
from turtle import shape
import xarray as xr
import pandas as pd
# import nctoolkit as nc
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import ListedColormap


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams.update({'font.size': 18})

# def visiual_online(path):
#     df = nc.open_data(path)
#     df.plot()

def interp_oxyformer2woa():
    woa_path = '/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/draw_figure/cimp5/nc_dir/woa.nc'
    # oxyformer_path = '/home/leadmove/dongliang/oxygen/DO_multidepth_v4_resplit_data/yearly_production/yearly_production.nc'
    oxyformer_path = '/home/leadmove/dongliang/oxygen/DO_sa_our_data/experiments/inference_monthly/mask_do_content_2003-2020_chlmask_yearlymean.nc'

    woa = xr.open_dataset(woa_path)
    oxyformer = xr.open_dataset(oxyformer_path)#.mean(dim='time')

    woa_dataframe = woa.to_dataframe()
    woa_result = pd.DataFrame()
    columns = woa_dataframe.columns

    for col in columns:
        temp = woa_dataframe[col].reset_index()
        temp['depth'] = float(col)
        temp.columns = ['latitude', 'longitude', 'o2', 'depth']
        woa_result = pd.concat([woa_result, temp])
    woa = woa_result.set_index(['latitude', 'longitude', 'depth']).to_xarray()
    
    oxyformer = oxyformer.isel(time=0).rename({'lon':'longitude', 'lat':'latitude'})
    # oxyformer_dataframe = oxyformer.to_dataframe()
    # oxyformer_result = pd.DataFrame()
    # columns = oxyformer_dataframe.columns
    # for col in columns:
    #     temp = oxyformer_dataframe[col].reset_index()
    #     temp['depth'] = float(col)
    #     temp.columns = ['latitude', 'longitude', 'o2', 'depth']
    #     oxyformer_result = pd.concat([oxyformer_result, temp])
    # oxyformer = oxyformer_result.set_index(['latitude', 'longitude', 'depth']).to_xarray()
    oxyformer_interp = oxyformer.interp(latitude=woa.latitude, longitude=woa.longitude, depth=woa.depth, method='linear')
    woa.to_netcdf('woa.nc')
    oxyformer_interp.to_netcdf('oxyformer_interp_new.nc')


def draw_spatial_map(woa, oxyformer, depth_list):
    lats = woa.latitude.values
    lons = woa.longitude.values
    levels = np.linspace(0, 400, 20)

    temp = []
    for index, depth in enumerate(depth_list):
        woa_depth = woa.sel(depth=depth)
        oxyformer_depth = oxyformer.sel(depth=depth)
        woa_o2 = woa_depth.o2.values
        oxyformer_o2 = oxyformer_depth.o2.values
        temp.append({'depth':depth, 'woa': woa_o2})
        temp.append({'depth':depth, 'oxyformer': oxyformer_o2})
        
    name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    # name = ['(a) WOA', '(b) Oxyformer', '(c) WOA', '(d) Oxyformer',
    #         '(e) WOA', '(f) Oxyformer', '(d) WOA', '(h) Oxyformer',
    #         '(i) WOA', '(j) Oxyformer', '(k) WOA', '(l) Oxyformer']

    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))

    # lons, lats, times, data = sample_data_3d((12, 73, 145))

    # fig = plt.figure(figsize=(10, 17), tight_layout=True)#, tight_layout=True
    fig = plt.figure(figsize=(24, 10))#, tight_layout=True
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 4),
                    axes_pad=0.4,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.1,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode

    colors = ['#7f0000', '#b30000', '#d7301f', '#ef6548', '#fc8d59', 
              '#fdbb84', '#fdd49e', '#fee8c8', '#fff7ec'
              ]
    
    my_cmap = ListedColormap(colors, name="my_cmap")

    for i, (ax, na, sample) in enumerate(zip(axgr, name, temp)):
        ax.coastlines()
        if i % 2 == 0:
            data = sample['woa']
            sub_name = 'WOA'
        else:
            data = sample['oxyformer']
            sub_name = 'Oxyformer'

        p = ax.contourf(lons, lats, data,
                        transform=ccrs.PlateCarree(),
                        cmap=my_cmap, levels=levels)#tab20c YlOrBr_r
        # ax.add_feature(cfeature.LAND, color='#CBCCCA')
        ax.gridlines(linestyle='--', color='black', alpha = 0.5)
        ax.set_title(r'$\bf{'+ na +'}$. '+sub_name+' '+str(sample['depth'])+'m', y=-0.17)#
        plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0, wspace=0, hspace =0)

    axgr.cbar_axes[0].colorbar(p)
    plt.savefig(r'./spatial_correlation.png', bbox_inches='tight')


def cal_cosine_similarity(woa, oxyformer, depth_list):
    cos_func = lambda a, b: np.sum(a *b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for index, depth in enumerate(depth_list):
        woa_depth = woa.sel(depth=depth)
        oxyformer_depth = oxyformer.sel(depth=depth)
        woa_o2 = woa_depth.o2.values
        oxyformer_o2 = oxyformer_depth.o2.values
        mask = (~np.isnan(woa_o2)) & (~np.isnan(oxyformer_o2))
        woa_mask = woa_o2[mask]
        oxy_mask = oxyformer_o2[mask]
        # print(depth, cos_func(woa_mask, oxy_mask))
        print(depth, np.sqrt(r2_score(woa_mask, oxy_mask)))
        


def draw_spatial_map_diff(woa, oxyformer, depth_list):
    lats = woa.latitude.values
    lons = woa.longitude.values
    levels = np.linspace(-50, 50, 20)

    temp = []
    for index, depth in enumerate(depth_list):
        woa_depth = woa.sel(depth=depth)
        oxyformer_depth = oxyformer.sel(depth=depth)
        woa_o2 = woa_depth.o2.values
        oxyformer_o2 = oxyformer_depth.o2.values

        mask = np.isnan(woa_o2)
        oxyformer_o2[mask] = np.nan

        diff = woa_o2 - oxyformer_o2

        temp.append({'depth':depth, 'diff': diff})
        

    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))

    fig = plt.figure(figsize=(24, 10))#, tight_layout=True
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=0.4,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.1,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode

    colors = [
        '#d73027',
        '#f46d43',
        '#fdae61',
        '#fee090',
        '#ffffbf',
        '#e0f3f8',
        '#abd9e9',
        '#74add1',
        '#4575b4',
    ]

    my_cmap = ListedColormap(colors, name="my_cmap")

    for i, (ax, sample) in enumerate(zip(axgr, temp)):
        ax.coastlines()        
        data = sample['diff']
        p = ax.contourf(lons, lats, data,
                        transform=ccrs.PlateCarree(),
                        cmap=my_cmap, levels=levels)#, 
        ax.gridlines(linestyle='--', color='black', alpha = 0.5)
        ax.set_title(str(sample['depth'])+'m', y=-0.15)#
        plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0, wspace=0, hspace =0)

    axgr.cbar_axes[0].colorbar(p)
    # cb = plt.colorbar(p, pad=0.08, orientation='horizontal', shrink=0.8) #cax=position, 
    plt.savefig(r'./spatial_correlation_diff.png', bbox_inches='tight')



if __name__ == '__main__':
    # interp_oxyformer2woa()
    woa = xr.open_dataset('./woa.nc')
    oxyformer = xr.open_dataset('./oxyformer_interp_new.nc')
    # visiual_online('./oxyformer_interp.nc')
    draw_spatial_map(woa=woa, oxyformer=oxyformer, depth_list=[5, 100, 300, 700, 1000, 3000])
    draw_spatial_map_diff(woa=woa, oxyformer=oxyformer, depth_list=[5, 100, 300, 700, 1000, 3000])
    
    # cal_cosine_similarity(woa=woa, oxyformer=oxyformer, depth_list=[5, 100, 300, 700, 1000, 3000])
    


