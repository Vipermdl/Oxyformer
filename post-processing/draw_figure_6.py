import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from collections import defaultdict

import xarray as xr
import numpy as np
import pandas as pd
import cv2, os

import sys
sys.path.append('../')
from figure_2.draw_figure2 import my_func

sys.path.append('../../')
from lib.utils import usecols

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams.update({'font.size': 15})

from matplotlib.pyplot import MultipleLocator
from sklearn.linear_model import LinearRegression

# '#2F4858' sohtcbtm
# #A36393 sohtc700

colorlist = [
        '#4F7416', '#B187F6', '#D16986', '#8ACCEE', '#2F4858',
        '#425D99', '#A36393', '#985850', '#57423F', '#EF7A6D',
        '#F3EED9', '#ED7E3B', '#3168BE', '#D54753', '#705E8D'
    ]

def my_func(a): 
    mask = a == 0
    if np.all(mask): return np.nan
    return LinearRegression().fit(np.arange(2003, 2021)[~mask].reshape(-1, 1), a[~mask]).coef_

def cal_corr_info(mask_name, corr='personr'):
    do_tot = xr.open_dataset('./do_content_per_site_remap.nc')
    mask = xr.open_dataset('./mask/{}'.format(mask_name))
    do_tot_mask = do_tot.where(mask.__xarray_dataarray_variable__)#.mean(dim=['lon', 'lat'])
    feature = xr.open_dataset('/home/leadmove/disk/oxygen/substain_feature/{}_yearly.nc'.format(mask_name[:-3]))#.mean(dim=['lon', 'lat'])
    feature  = feature.isel(bnds=0).to_dataframe().dropna().reset_index()
    label = do_tot_mask.__xarray_dataarray_variable__.to_dataframe().dropna().reset_index()
    feature['time'] = feature['time'].apply(lambda x: str(pd.to_datetime(x).year))
    label['time'] = label['time'].apply(lambda x: str(pd.to_datetime(x).year))
    data = pd.merge(feature, label, on=['time', 'lat', 'lon'])
    data = data.groupby(['time']).mean()
    if corr == 'mutual_info':
        infos = mutual_info_regression(data[usecols].values, data['__xarray_dataarray_variable__'].values)
    elif corr == 'personr': infos = [pearsonr(data[uc].values, data['__xarray_dataarray_variable__'].values)[0] for uc in usecols]
    return infos

def strided_app(a, L, S): # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

def numpyEWMA(price, windowSize):
    weights = np.exp(np.linspace(-1., 0., windowSize))
    weights /= weights.sum()
    a2D = strided_app(price, windowSize, 1)
    returnArray = np.empty((price.shape[0]))
    returnArray.fill(np.nan)
    for index in (range(a2D.shape[0])):
        returnArray[index + windowSize-1] = np.convolve(weights, a2D[index])[windowSize - 1:-windowSize + 1]
    return np.where(np.isnan(returnArray), price, returnArray)
    # return np.reshape(returnArray, (-1, 1))

def get_feature_trend(mask_name, root_path = '/home/leadmove/disk/oxygen/substain_feature/'):
    do_tot = xr.open_dataset('./do_content_per_site_remap.nc')
    mask = xr.open_dataset('./mask/{}'.format(mask_name))
    do_tot_mask = do_tot.where(mask.__xarray_dataarray_variable__).mean(dim=['lon', 'lat'])
    filename = mask_name[:-3]+'_yearly.nc'
    data = xr.open_dataset(os.path.join(root_path, filename))
    data = data.mean(dim=['lon', 'lat'])
    result = {}
    cols = np.array(usecols)

    val = do_tot_mask.__xarray_dataarray_variable__.values

    print(mask_name, my_func(val))   

    # val = (val - val.min()) / (val.max() - val.min())
    result['o2'] = pd.DataFrame({'temp': val}).ewm(span=6).mean().values.reshape(-1)
    for var in cols: 
        val = data[var].values
        # val = (val - val.min()) / (val.max() - val.min())
        result[var] = pd.DataFrame({'temp': val}).ewm(span=6).mean().values.reshape(-1)
    return result

def draw_feature_trend():
    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(4, 9, wspace=0.5, hspace=0.4)
    ax = plt.subplot(grid[:3, :3], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=32, s=r'$\bf{c}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=15, s=r'$\bf{d}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())

    name = ['chlor_a', 'sosaline', 'sohtcbtm', 'sohefldo', 'u10', 'mslh']
    color = [colorlist[usecols.index(n)] for n in name] + ['#FF0000',]
    name = name + ['$O_2$',]

    ax2 = plt.subplot(grid[3, 0])
    ax2.barh(name[:3], [0.3]*3, color=color[:3], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(name[:3], [0.5]*3)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index+0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(-0.5, 4)
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')
    ax2 = plt.subplot(grid[3, 1])
    ax2.barh(name[3:5], [0.3]*2, color=color[3:5], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(name[3:5], [0.5]*2)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(-0.5, 4)
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')
    ax2 = plt.subplot(grid[3, 2])
    ax2.barh(name[5:], [0.3]*2, color=color[5:], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(name[5:], [0.5]*2)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(-0.5, 4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')


    ax4 = plt.subplot(grid[:2, 3:6])
    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='Equatorial_Indian_Ocean.nc')
    colors = np.array(colorlist) #[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm',] + [maxcol]: continue
        if var == 'o2': linewidth=4
        else: linewidth=2
        ax4.plot(np.arange(2003, 2021), val, label=var, color=color, linewidth=linewidth, marker="v")
    ax4.set_yticks(np.arange(0, 1.2, 0.25))
    ax4.set_xticks(np.arange(2003, 2020)[::4])
    ax4.grid()
    ax4.set_title(r'$\bf{a.}$', fontsize=28, x=0.92, y=0.85, color='red')


    ax5 = plt.subplot(grid[:2, 6:9])
    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='South_Pacific_Ocean.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm'] + [maxcol]: continue
        if var == 'o2': linewidth=4
        else: linewidth=2
        ax5.plot(np.arange(2003, 2021), val, label=var, color=color, linewidth=linewidth, marker="v")
    # ax5.set_yticks([])
    ax5.set_yticks(np.arange(0, 1.2, 0.25))
    ax5.set_xticks(np.arange(2003, 2020)[::4])
    ax5.grid()
    ax5.set_title(r'$\bf{b.}$', fontsize=28, x=0.92, y=0.85, color='red')


    ax6 = plt.subplot(grid[2:, 3:6], )
    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='North_Atlantic_ocean.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm']+[maxcol]: continue
        if var == 'o2': linewidth=4
        else: linewidth=2
        ax6.plot(np.arange(2003, 2021), val, label=var, color=color, linewidth=linewidth, marker="v")
    ax6.set_yticks(np.arange(0, 1.2, 0.25))
    ax6.set_xticks(np.arange(2003, 2020)[::4])
    ax6.grid()
    ax6.set_title(r'$\bf{c.}$', fontsize=28, x=0.92, y=0.85, color='blue')


    ax7 = plt.subplot(grid[2:, 6:9], )
    infos = cal_corr_info(mask_name='california_west_coast.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='california_west_coast.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm']+[maxcol]: continue
        if var == 'o2': linewidth=4
        else: linewidth=2
        ax7.plot(np.arange(2003, 2021), val, label=var, color=color, linewidth=linewidth, marker="v")
    # ax7.set_yticks([])
    ax7.set_yticks(np.arange(0, 1.2, 0.25))
    ax7.set_xticks(np.arange(2003, 2020)[::4])
    ax7.grid()
    ax7.set_title(r'$\bf{d.}$', fontsize=28, x=0.92, y=0.85, color='blue')

    plt.savefig('a.jpg', bbox_inches='tight')

def draw_mutual_info_bar():
    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(4, 7, wspace=0.4, hspace=0.3)
    ax = plt.subplot(grid[:3, :3], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    # ax.text(s=r'$\bf{b}$', fontsize=28, x= 0.5, y=0.7, color='red')

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=34, s=r'$\bf{c}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=18, s=r'$\bf{d}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    
    ax2 = plt.subplot(grid[3, 0])
    ax2.barh(usecols[:5], [0.3]*5, color=colorlist[:5], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[:5], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index+0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 1])
    ax2.barh(usecols[5:10], [0.3]*5, color=colorlist[5:10], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[5:10], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
        
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 2])
    ax2.barh(usecols[10:], [0.3]*5, color=colorlist[10:], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[10:], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
        
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[:2, 3:5])
    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(0, 1)
    ax2.set_title(r'$\bf{a.}$', fontsize=28, x=0.05, y=0.85, color='red')

    ax4 = plt.subplot(grid[:2, 5:7], )
    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc')
    ax4.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax4.set_xticks([])
    ax4.set_ylim(0, 1)
    ax4.set_title(r'$\bf{b.}$', fontsize=28, x=0.05, y=0.85, color='red')

    ax6 = plt.subplot(grid[2:, 3:5], )
    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc')
    ax6.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax6.set_xticks([])
    ax6.set_ylim(0, 1)
    ax6.set_title(r'$\bf{c.}$', fontsize=28, x=0.05, y=0.85, color='blue')

    ax8 = plt.subplot(grid[2:, 5:7], )
    infos = cal_corr_info(mask_name='california_west_coast.nc')
    ax8.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax8.set_xticks([])
    ax8.set_ylim(0, 1)
    ax8.set_title(r'$\bf{d.}$', fontsize=28, x=0.05, y=0.85, color='blue')

    plt.savefig('figure_4_mutual_bar.jpg', bbox_inches='tight')

def draw_pearconr_info_bar():
    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(4, 7, wspace=0.4, hspace=0.3)
    ax = plt.subplot(grid[:3, :3], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    # ax.text(s=r'$\bf{b}$', fontsize=28, x= 0.5, y=0.7, color='red')

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=34, s=r'$\bf{c}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=18, s=r'$\bf{d}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    
    ax2 = plt.subplot(grid[3, 0])
    ax2.barh(usecols[:5], [0.3]*5, color=colorlist[:5], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[:5], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index+0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 1])
    ax2.barh(usecols[5:10], [0.3]*5, color=colorlist[5:10], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[5:10], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
        
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 2])
    ax2.barh(usecols[10:], [0.3]*5, color=colorlist[10:], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[10:], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.65, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
        
    ax2.set_xlim(-0.2, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[:2, 3:5])
    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{a.}$', fontsize=28, x=0.92, y=0.85, color='red')

    ax4 = plt.subplot(grid[:2, 5:7], )
    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc')
    ax4.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax4.set_xticks([])
    ax4.set_ylim(-1, 1)
    ax4.set_title(r'$\bf{b.}$', fontsize=28, x=0.92, y=0.85, color='red')

    ax6 = plt.subplot(grid[2:, 3:5], )
    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc')
    ax6.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax6.set_xticks([])
    ax6.set_ylim(-1, 1)
    ax6.set_title(r'$\bf{c.}$', fontsize=28, x=0.92, y=0.85, color='blue')

    ax8 = plt.subplot(grid[2:, 5:7], )
    infos = cal_corr_info(mask_name='california_west_coast.nc')
    ax8.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax8.set_xticks([])
    ax8.set_ylim(-1, 1)
    ax8.set_title(r'$\bf{d.}$', fontsize=28, x=0.92, y=0.85, color='blue')

    plt.savefig('figure_4_personr_bar.jpg', bbox_inches='tight')

def get_region_do_concentration(temp_mask):
    temp = temp_mask.o2.median(dim=['lat', 'lon'])
    lower_bound = temp_mask.o2.quantile(0.25, dim=['lat', 'lon'])
    upper_bound = temp_mask.o2.quantile(0.75, dim=['lat', 'lon'])
    depth_300 = temp.sel(depth=300).values.squeeze()
    depth_1000 = temp.sel(depth=1000).values.squeeze()
    depth_3000 = temp.sel(depth=3000).values.squeeze()
    lower_300 = lower_bound.sel(depth=300).values.squeeze()
    lower_1000 = lower_bound.sel(depth=1000).values.squeeze()
    lower_3000 = lower_bound.sel(depth=3000).values.squeeze()
    upper_300 = upper_bound.sel(depth=300).values.squeeze()
    upper_1000 = upper_bound.sel(depth=1000).values.squeeze()
    upper_3000 = upper_bound.sel(depth=3000).values.squeeze()
    mask_300 = np.isnan(depth_300) | np.isnan(lower_300) | np.isnan(upper_300)
    mask_1000 = np.isnan(depth_1000) | np.isnan(lower_1000) | np.isnan(upper_1000)
    mask_3000 = np.isnan(depth_3000) | np.isnan(lower_3000) | np.isnan(upper_3000)
    depth_300 = depth_300[~mask_300]
    depth_1000 = depth_1000[~mask_1000]
    depth_3000 = depth_3000[~mask_3000]

    lower_300 = lower_300[~mask_300]
    lower_1000 = lower_1000[~mask_1000]
    lower_3000 = lower_3000[~mask_3000]
    upper_300 = upper_300[~mask_300]
    upper_1000 = upper_1000[~mask_1000]
    upper_3000 = upper_3000[~mask_3000]
    lower_300 = depth_300 - lower_300
    lower_1000 = depth_1000 - lower_1000
    lower_3000 = depth_3000 - lower_3000
    upper_300 = upper_300 - depth_300
    upper_1000 = upper_1000 - depth_1000
    upper_3000 = upper_3000 - depth_3000

    yeer_300 = np.stack([lower_300, upper_300])
    yeer_1000 = np.stack([lower_1000, upper_1000])
    yeer_3000 = np.stack([lower_3000, upper_3000])
    return depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000

def draw_mutual_info_bar_do_content():
    fig = plt.figure(figsize=(24, 16))
    grid = plt.GridSpec(8, 9, wspace=0.4, hspace=0.3)
    ax = plt.subplot(grid[:3, :3], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    dataset = xr.open_dataset('region_mask.nc')
    x_major_locator = MultipleLocator(3)
    y_major_locator = MultipleLocator(10)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    # ax.text(s=r'$\bf{b}$', fontsize=28, x= 0.5, y=0.7, color='red')

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=32, s=r'$\bf{c}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=15, s=r'$\bf{d}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    
    ax2 = plt.subplot(grid[3, 0])
    ax2.barh(usecols[:5], [0.3]*5, color=colorlist[:5], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[:5], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index+0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 1])
    ax2.barh(usecols[5:10], [0.3]*5, color=colorlist[5:10], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[5:10], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 2])
    ax2.barh(usecols[10:], [0.3]*5, color=colorlist[10:], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[10:], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.75, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[:2, 3:6])
    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc', corr='personr')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{a.}$', fontsize=28, x=-0.12, y=0.85, color='red')

    ax3 = plt.subplot(grid[:2, 6:9])
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('Equatorial_Indian_Ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax3.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax3.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax3.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax3.set_xticks(np.arange(2003, 2020)[::4])
    ax3.legend(loc=0)
    ax3.grid()
    ax3.set_ylim(-30, 30)
    ax3.xaxis.set_major_locator(x_major_locator)
    ax3.yaxis.set_major_locator(y_major_locator)


    ax4 = plt.subplot(grid[2:4, 3:6], )
    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc', corr='personr')
    ax4.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax4.set_xticks([])
    ax4.set_ylim(-1, 1)
    ax4.set_title(r'$\bf{b.}$', fontsize=28, x=-0.12, y=0.85, color='red')

    ax5 = plt.subplot(grid[2:4, 6:9], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('South_Pacific_Ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax5.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax5.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax5.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax5.set_xticks(np.arange(2003, 2020)[::4])
    ax5.legend(loc=0)
    ax5.grid()
    ax5.set_ylim(-30, 30)
    ax5.xaxis.set_major_locator(x_major_locator)
    ax5.yaxis.set_major_locator(y_major_locator)

    ax6 = plt.subplot(grid[4:6, 3:6], )
    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc', corr='personr')
    ax6.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax6.set_xticks([])
    ax6.set_ylim(-1, 1)
    ax6.set_title(r'$\bf{c.}$', fontsize=28, x=-0.12, y=0.85, color='blue')

    ax7 = plt.subplot(grid[4:6, 6:9], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('North_Atlantic_ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax7.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax7.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax7.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax7.set_xticks(np.arange(2003, 2020)[::4])
    ax7.legend(loc=0)
    ax7.grid()
    ax7.set_ylim(-30, 30)
    ax7.xaxis.set_major_locator(x_major_locator)
    ax7.yaxis.set_major_locator(y_major_locator)

    ax8 = plt.subplot(grid[6:, 3:6], )
    infos = cal_corr_info(mask_name='california_west_coast.nc', corr='personr')
    ax8.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax8.set_xticks([])
    ax8.set_ylim(-1, 1)
    ax8.set_title(r'$\bf{d.}$', fontsize=28, x=-0.12, y=0.85, color='blue')

    ax9 = plt.subplot(grid[6:, 6:9], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('california_west_coast.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax9.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax9.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax9.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax9.set_xticks(np.arange(2003, 2020)[::4])
    ax9.legend(loc=0)
    ax9.grid()
    ax9.set_ylim(-30, 30)
    ax9.xaxis.set_major_locator(x_major_locator)
    ax9.yaxis.set_major_locator(y_major_locator)

    plt.savefig('figure_4_mutual_bar_do_content.jpg', bbox_inches='tight')

def get_map():
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
    ax = plt.subplot(grid[:, :], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    dataset = xr.open_dataset('region_mask.nc')
    x_major_locator = MultipleLocator(3)
    y_major_locator = MultipleLocator(10)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    # ax.text(s=r'$\bf{b}$', fontsize=28, x= 0.5, y=0.7, color='red')

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=28,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=32, s=r'$\bf{c}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=15, s=r'$\bf{d}$', fontsize=28,  color='blue', transform=ccrs.PlateCarree())

    plt.savefig('map.jpg', bbox_inches='tight')
    
def new_draw_feature_trend_a():
    fig = plt.figure(figsize=(8, 4))

    name = ['chlor_a', 'sosaline', 'sohtcbtm', 'sohefldo', 'u10', 'mslh']
    color = [colorlist[usecols.index(n)] for n in name] + ['#FF0000',]
    name = name + ['$O_2$',]

    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    result = get_feature_trend(mask_name='Equatorial_Indian_Ocean.nc')
    colors = np.array(colorlist) #[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    result_plot = defaultdict(list)

    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm',] + [maxcol]: continue
        result_plot[var].append([val, color])
        
    ax_cof = HostAxes(fig, [0, 0, 0.9, 0.9])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    #parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

    #append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)

    #invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(90,0))
    ax_wear.axis['right3'] = cp_axisline(loc='right', axes=ax_wear, offset=(180,0))
    ax_cp.axis['right4'] = wear_axisline(loc='right', axes=ax_cp, offset=(270,0))

    fig.add_axes(ax_cof)

    #set label for axis
    ax_cof.set_ylabel(r'$O_2 (mol \cdot m^{-2})$')
    ax_cof.set_xlabel('Year')
    ax_temp.set_ylabel('chlorophyll-a concentration \n ($mg \cdot m^{-3}$)')
    ax_load.set_ylabel('sea surface salinity \n ($PSU$)')
    ax_cp.set_ylabel('net downward heat flux \n ($W \cdot m^{-2}$)')
    ax_wear.set_ylabel('OHC for total water column \n ($10^{10} J \cdot m^{-2}$)')

    curve_cof, = ax_cof.plot(np.arange(2003, 2021), result_plot['o2'][0][0], color=result_plot['o2'][0][1], marker="v", linewidth=4)
    curve_temp, = ax_temp.plot(np.arange(2003, 2021), result_plot['chlor_a'][0][0], color=result_plot['chlor_a'][0][1], marker="v", linewidth=3)
    curve_load, = ax_load.plot(np.arange(2003, 2021), result_plot['sosaline'][0][0], color=result_plot['sosaline'][0][1], marker="v", linewidth=3)
    curve_cp, = ax_cp.plot(np.arange(2003, 2021), result_plot['sohefldo'][0][0], color=result_plot['sohefldo'][0][1], marker="v", linewidth=3)
    curve_wear, = ax_wear.plot(np.arange(2003, 2021), result_plot['sohtcbtm'][0][0]/ 10**10, color=result_plot['sohtcbtm'][0][1], marker="v", linewidth=3)

    ax_cof.set_xlim(2002.8, 2020.2)

    # ax_cof.set_xticks(np.arange(2004, 2021, 2), fontsize=12)
    
    ax_cof.set_yticks(np.arange(655, 740, 15))
    ax_temp.set_yticks(np.arange(0.15, 0.2, 0.01))
    ax_load.set_yticks(np.arange(34.96, 35.16, 0.04))
    ax_cp.set_yticks(np.arange(20, 42, 5))
    ax_wear.set_yticks(np.arange(7.74, 7.94, 0.04))

    
    # ax_cof.legend()

    ax_cof.axis['left'].label.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].label.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].label.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].label.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].label.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticks.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticks.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticks.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticks.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].major_ticks.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticklabels.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticklabels.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticklabels.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticklabels.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].major_ticklabels.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].line.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].line.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].line.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].line.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].line.set_color(result_plot['sohtcbtm'][0][1])
    # ax_cof.set_title(r'$\bf{a.}$', fontsize=28, x=-0.1, y=0.95, color='red')

    plt.savefig('Equatorial_Indian_Ocean.jpg', bbox_inches='tight')

def new_draw_feature_trend_b():
    fig = plt.figure(figsize=(8, 4))

    name = ['chlor_a', 'sosaline', 'sohtcbtm', 'sohefldo', 'u10', 'mslh']
    color = [colorlist[usecols.index(n)] for n in name] + ['#FF0000',]
    name = name + ['$O_2$',]

    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='South_Pacific_Ocean.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    
    result_plot = defaultdict(list)

    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm'] + [maxcol]: continue
        result_plot[var].append([val, color])    
    
    ax_cof = HostAxes(fig, [0, 0, 0.9, 0.9])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    #parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

    #append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)

    #invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(90,0))
    ax_wear.axis['right3'] = cp_axisline(loc='right', axes=ax_wear, offset=(180,0))
    ax_cp.axis['right4'] = wear_axisline(loc='right', axes=ax_cp, offset=(270,0))

    fig.add_axes(ax_cof)

    ax_cof.set_xlabel('Year')
    
    #set label for axis
    ax_cof.set_ylabel(r'$O_2 (mol \cdot m^{-2})$')
    ax_cof.set_xlabel('Year')
    ax_temp.set_ylabel('chlorophylla-a concentration \n ($mg \cdot m^{-3}$)')
    ax_load.set_ylabel('sea surface salinity \n ($PSU$)')
    ax_cp.set_ylabel('surface latent heat flux \n ($W \cdot m^{-2}$)')
    ax_wear.set_ylabel('OHC for total water column \n ($10^{10} J \cdot m^{-2}$)')

    curve_cof, = ax_cof.plot(np.arange(2003, 2021), result_plot['o2'][0][0], color=result_plot['o2'][0][1], marker="v", linewidth=4)
    curve_temp, = ax_temp.plot(np.arange(2003, 2021), result_plot['chlor_a'][0][0], color=result_plot['chlor_a'][0][1], marker="v", linewidth=3)
    curve_load, = ax_load.plot(np.arange(2003, 2021), result_plot['sosaline'][0][0], color=result_plot['sosaline'][0][1], marker="v", linewidth=3)
    curve_cp, = ax_cp.plot(np.arange(2003, 2021), result_plot['mslh'][0][0], color=result_plot['mslh'][0][1], marker="v", linewidth=3)
    curve_wear, = ax_wear.plot(np.arange(2003, 2021), result_plot['sohtcbtm'][0][0]/ 10**10, color=result_plot['sohtcbtm'][0][1], marker="v", linewidth=3)

    ax_cof.set_xlim(2002.8, 2020.2)

    ax_cof.set_yticks(np.arange(635, 720, 15))
    ax_temp.set_yticks(np.arange(0.10, 0.16, 0.01))
    ax_load.set_yticks(np.arange(34.40, 34.56, 0.04))
    ax_cp.set_yticks(np.arange(-95.00, -90.00, 0.90))
    ax_wear.set_yticks(np.arange(5.32, 5.43, 0.02))


    ax_cof.axis['left'].label.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].label.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].label.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].label.set_color(result_plot['mslh'][0][1])
    ax_wear.axis['right3'].label.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticks.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticks.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticks.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticks.set_color(result_plot['mslh'][0][1])
    ax_wear.axis['right3'].major_ticks.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticklabels.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticklabels.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticklabels.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticklabels.set_color(result_plot['mslh'][0][1])
    ax_wear.axis['right3'].major_ticklabels.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].line.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].line.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].line.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].line.set_color(result_plot['mslh'][0][1])
    ax_wear.axis['right3'].line.set_color(result_plot['sohtcbtm'][0][1])
    # ax_cof.set_title(r'$\bf{b.}$', fontsize=28, x=-0.1, y=0.95, color='red')
    plt.savefig('South_Pacific_Ocean.jpg', bbox_inches='tight')

def new_draw_feature_trend_c():
    fig = plt.figure(figsize=(8, 4))

    name = ['chlor_a', 'sosaline', 'sohtcbtm', 'sohefldo', 'u10', 'mslh']
    color = [colorlist[usecols.index(n)] for n in name] + ['#FF0000',]
    name = name + ['$O_2$',]

    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='North_Atlantic_ocean.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    result_plot = defaultdict(list)
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm']+[maxcol]: continue
        result_plot[var].append([val, color])
    
    ax_cof = HostAxes(fig, [0, 0, 0.9, 0.9])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    #parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

    #append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)

    #invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(90,0))
    ax_wear.axis['right3'] = cp_axisline(loc='right', axes=ax_wear, offset=(180,0))
    ax_cp.axis['right4'] = wear_axisline(loc='right', axes=ax_cp, offset=(270,0))

    fig.add_axes(ax_cof)

    #set label for axis
    ax_cof.set_ylabel(r'$O_2 (mol \cdot m^{-2})$')
    ax_cof.set_xlabel('Year')
    ax_temp.set_ylabel('chlorophyll-a concentration \n ($mg \cdot m^{-3}$)')
    ax_load.set_ylabel('sea surface salinity \n ($PSU$)')
    ax_cp.set_ylabel('10 m U wind \n ($m \cdot s^{-1}$)')
    ax_wear.set_ylabel('OHC for total water column \n ($10^{10} J \cdot m^{-2}$)')


    curve_cof, = ax_cof.plot(np.arange(2003, 2021), result_plot['o2'][0][0], color=result_plot['o2'][0][1], marker="v", linewidth=4)
    curve_temp, = ax_temp.plot(np.arange(2003, 2021), result_plot['chlor_a'][0][0], color=result_plot['chlor_a'][0][1], marker="v", linewidth=3)
    curve_load, = ax_load.plot(np.arange(2003, 2021), result_plot['sosaline'][0][0], color=result_plot['sosaline'][0][1], marker="v", linewidth=3)
    curve_cp, = ax_cp.plot(np.arange(2003, 2021), result_plot['u10'][0][0], color=result_plot['u10'][0][1], marker="v", linewidth=3)
    curve_wear, = ax_wear.plot(np.arange(2003, 2021), result_plot['sohtcbtm'][0][0]/ 10**10, color=result_plot['sohtcbtm'][0][1], marker="v", linewidth=3)

    ax_cof.set_xlim(2002.8, 2020.2)

    ax_cof.set_yticks(np.arange(940, 980, 8))
    ax_temp.set_yticks(np.arange(0.36, 0.62, 0.05))
    ax_load.set_yticks(np.arange(35.22, 35.42, 0.05))
    ax_cp.set_yticks(np.arange(2.9, 4.4, 0.3))
    ax_wear.set_yticks(np.arange(6.93, 7.08, 0.03))

    ax_cof.axis['left'].label.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].label.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].label.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].label.set_color(result_plot['u10'][0][1])
    ax_wear.axis['right3'].label.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticks.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticks.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticks.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticks.set_color(result_plot['u10'][0][1])
    ax_wear.axis['right3'].major_ticks.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticklabels.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticklabels.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticklabels.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticklabels.set_color(result_plot['u10'][0][1])
    ax_wear.axis['right3'].major_ticklabels.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].line.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].line.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].line.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].line.set_color(result_plot['u10'][0][1])
    ax_wear.axis['right3'].line.set_color(result_plot['sohtcbtm'][0][1])
    # ax_cof.set_title(r'$\bf{c.}$', fontsize=28, x=-0.1, y=0.95, color='blue')
    plt.savefig('North_Atlantic_ocean.jpg', bbox_inches='tight')

def new_draw_feature_trend_d():
    fig = plt.figure(figsize=(8, 4))

    name = ['chlor_a', 'sosaline', 'sohtcbtm', 'sohefldo', 'u10', 'mslh']
    color = [colorlist[usecols.index(n)] for n in name] + ['#FF0000',]
    name = name + ['$O_2$',]

    infos = cal_corr_info(mask_name='california_west_coast.nc', corr='personr')
    infos = np.abs(infos)
    maxcol = np.array(usecols)[[3, 4, 6, 8, 9, 10, 12, 13, 14]][np.nanargmax(infos[[3, 4, 6, 8, 9, 10, 12, 13, 14]])]
    print(maxcol)
    result = get_feature_trend(mask_name='california_west_coast.nc')
    colors = np.array(colorlist)#[infos > 0.6]
    colors = np.insert(colors, 0, '#FF0000')
    result_plot = defaultdict(list)
    for color, (var, val) in zip(colors, result.items()):
        if var not in ['o2', 'chlor_a', 'sosaline', 'sohtcbtm']+[maxcol]: continue
        result_plot[var].append([val, color])

    ax_cof = HostAxes(fig, [0, 0, 0.9, 0.9])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    #parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

    #append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)

    #invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(90,0))
    ax_wear.axis['right3'] = cp_axisline(loc='right', axes=ax_wear, offset=(180,0))
    ax_cp.axis['right4'] = wear_axisline(loc='right', axes=ax_cp, offset=(270,0))

    fig.add_axes(ax_cof)

    ax_cof.set_ylabel(r'$O_2 (mol \cdot m^{-2})$')
    ax_cof.set_xlabel('Year')
    ax_temp.set_ylabel('chlorophyll-a concentration \n ($mg \cdot m^{-3}$)')
    ax_load.set_ylabel('sea surface salinity \n ($PSU$)')
    ax_cp.set_ylabel('net downward heat flux \n ($W \cdot m^{-2}$)')
    ax_wear.set_ylabel('OHC for total water column \n ($10^{10} J \cdot m^{-2}$)')

    curve_cof, = ax_cof.plot(np.arange(2003, 2021), result_plot['o2'][0][0], color=result_plot['o2'][0][1], marker="v", linewidth=4)
    curve_temp, = ax_temp.plot(np.arange(2003, 2021), result_plot['chlor_a'][0][0], color=result_plot['chlor_a'][0][1], marker="v", linewidth=3)
    curve_load, = ax_load.plot(np.arange(2003, 2021), result_plot['sosaline'][0][0], color=result_plot['sosaline'][0][1], marker="v", linewidth=3)
    curve_cp, = ax_cp.plot(np.arange(2003, 2021), result_plot['sohefldo'][0][0], color=result_plot['sohefldo'][0][1], marker="v", linewidth=3)
    curve_wear, = ax_wear.plot(np.arange(2003, 2021), result_plot['sohtcbtm'][0][0]/ 10**10, color=result_plot['sohtcbtm'][0][1], marker="v", linewidth=3)

    ax_cof.set_xlim(2002.8, 2020.2)
    # ax_cof.legend()
    # ax_cof.set_ylim(2.90, 3.28)
    # ax_temp.set_ylim(0.066, 0.076)
    # ax_load.set_ylim(35.30, 35.55)
    # ax_cp.set_ylim(1.05, 1.30)
    # ax_wear.set_ylim(2002.8, 2020.2)
    ax_cof.set_yticks(np.arange(430, 490, 10))
    ax_temp.set_yticks(np.arange(0.069, 0.081, 0.002))
    ax_load.set_yticks(np.arange(33.80, 34.24, 0.08))
    ax_cp.set_yticks(np.arange(-2, 14, 3.5))
    ax_wear.set_yticks(np.arange(5.8, 5.94, 0.03))

    ax_cof.axis['left'].label.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].label.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].label.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].label.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].label.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticks.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticks.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticks.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticks.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].major_ticks.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].major_ticklabels.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].major_ticklabels.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].major_ticklabels.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].major_ticklabels.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].major_ticklabels.set_color(result_plot['sohtcbtm'][0][1])

    ax_cof.axis['left'].line.set_color(result_plot['o2'][0][1])
    ax_temp.axis['right'].line.set_color(result_plot['chlor_a'][0][1])
    ax_load.axis['right2'].line.set_color(result_plot['sosaline'][0][1])
    ax_cp.axis['right4'].line.set_color(result_plot['sohefldo'][0][1])
    ax_wear.axis['right3'].line.set_color(result_plot['sohtcbtm'][0][1])
    # ax_cof.set_title(r'$\bf{d.}$', fontsize=28, x=-0.1, y=0.95, color='blue')
    plt.savefig('california_west_coast.jpg', bbox_inches='tight')

def draw_head():
    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(4, 6, wspace=0.1, hspace=0)
    ax = plt.subplot(grid[:, :5], projection=ccrs.Mollweide(central_longitude=-155))
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#CBCCCA')
    ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    dataset = xr.open_dataset('region_mask.nc')
    x_major_locator = MultipleLocator(3)
    y_major_locator = MultipleLocator(10)

    california_west_coast_mask = xr.open_dataset('mask/california_west_coast.nc')
    california_west_coast_mask = california_west_coast_mask.to_dataframe().reset_index()
    california_west_coast_mask = california_west_coast_mask[california_west_coast_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(california_west_coast_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=california_west_coast_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/'))#hatch='/'
    Equatorial_Indian_Ocean_mask = xr.open_dataset('mask/Equatorial_Indian_Ocean.nc')
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask.to_dataframe().reset_index()
    Equatorial_Indian_Ocean_mask = Equatorial_Indian_Ocean_mask[Equatorial_Indian_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(Equatorial_Indian_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=Equatorial_Indian_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    # ax.text(s=r'$\bf{b}$', fontsize=28, x= 0.5, y=0.7, color='red')

    North_Atlantic_ocean_mask = xr.open_dataset('mask/North_Atlantic_ocean.nc')
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask.to_dataframe().reset_index()
    North_Atlantic_ocean_mask = North_Atlantic_ocean_mask[North_Atlantic_ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(North_Atlantic_ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=North_Atlantic_ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='blue',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    
    South_Pacific_Ocean_mask = xr.open_dataset('mask/South_Pacific_Ocean.nc')
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask.to_dataframe().reset_index()
    South_Pacific_Ocean_mask = South_Pacific_Ocean_mask[South_Pacific_Ocean_mask['__xarray_dataarray_variable__']]
    hull = ConvexHull(South_Pacific_Ocean_mask[['lon', 'lat']].values)
    ax.add_patch(mpatches.Polygon(xy=South_Pacific_Ocean_mask[['lon', 'lat']].values[hull.vertices], alpha=0.4, facecolor='red',
                                transform=ccrs.PlateCarree(), edgecolor='black',  hatch='/',))#hatch='/', linewidth=1
    ax.text(x=75, y=-4, s=r'$\bf{a}$', fontsize=48,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-132, y=-25, s=r'$\bf{b}$', fontsize=48,  color='red', transform=ccrs.PlateCarree())
    ax.text(x=-43, y=32, s=r'$\bf{c}$', fontsize=48,  color='blue', transform=ccrs.PlateCarree())
    ax.text(x=-147, y=15, s=r'$\bf{d}$', fontsize=48,  color='blue', transform=ccrs.PlateCarree())
    
    height = 0.6

    ax2 = plt.subplot(grid[1, 5])
    ax2.barh(usecols[:5], [0.3]*5, color=colorlist[:5], linewidth=2, height=height, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[:5], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.6, 0.2 + index+0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[2, 5])
    ax2.barh(usecols[5:10], [0.3]*5, color=colorlist[5:10], linewidth=2, height=height, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[5:10], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.6, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')

    ax2 = plt.subplot(grid[3, 5])
    ax2.barh(usecols[10:], [0.3]*5, color=colorlist[10:], linewidth=2, height=height, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[10:], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.6, 0.2 + index + 0.1, x, ha='center', va='top', fontsize=14)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['right', 'top', 'left', 'bottom']].set_color('none')
    
    plt.savefig('head.jpg', bbox_inches='tight')

def draw_colorbar():
    fig = plt.figure(figsize=(21, 4))
    grid = plt.GridSpec(1, 3, wspace=0.08, hspace=0)

    usecols = [
        'chlorophyll-a concentration',
        'sea surface salinity',
        'OHC for upper 700 m',
        'mixed layer depth 0.01',
        'net downward heat flux',
        'OHC for total water column',
        'mixed layer depth 0.03',
        'OHC for upper 300 m',
        '10 m V wind',
        '10 m U wind',
        'air pressure at mean sea level',
        'sea surface temperature',
        'surface latent heat flux',
        'surface net short-wave radiation flux',
        'depth of water',
    ]

    ax2 = plt.subplot(grid[0, 0])
    ax2.barh(usecols[0::3][::-1], [0.12]*5, color=colorlist[0::3][::-1], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[0::3][::-1], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.6, 0.2 + index, x, ha='center', va='top', fontsize=18)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for x in ['right', 'top', 'left', 'bottom']:
        ax2.spines[x].set_color('none')

    ax2 = plt.subplot(grid[0, 1])
    ax2.barh(usecols[1::3][::-1], [0.12]*5, color=colorlist[1::3][::-1], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[1::3][::-1], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.7, 0.2 + index, x, ha='center', va='top', fontsize=18)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for x in ['right', 'top', 'left', 'bottom']:
        ax2.spines[x].set_color('none')

    ax2 = plt.subplot(grid[0, 2])
    ax2.barh(usecols[2::3][::-1], [0.12]*5, color=colorlist[2::3][::-1], linewidth=2, **{'edgecolor': 'black'})
    for index, (x, y) in enumerate(zip(usecols[2::3][::-1], [0.5]*5)): # ha: horizontal alignment # va: vertical alignment
        ax2.text(0.6, 0.2 + index, x, ha='center', va='top', fontsize=18)
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for x in ['right', 'top', 'left', 'bottom']:
        ax2.spines[x].set_color('none')

    plt.savefig('colorbar1.jpg', bbox_inches='tight')

def new_corr_a():
    fig = plt.figure(figsize=(8, 4))
    grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[0, 0])
    infos = cal_corr_info(mask_name='Equatorial_Indian_Ocean.nc', corr='personr')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{a.}$', fontsize=28, x=-0.12, y=0.85, color='red')
    plt.savefig('Equatorial_Indian_Ocean_corr.jpg', bbox_inches='tight')

def new_corr_b():
    fig = plt.figure(figsize=(8, 4))
    grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[0, 0])
    infos = cal_corr_info(mask_name='South_Pacific_Ocean.nc', corr='personr')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{b.}$', fontsize=28, x=-0.12, y=0.85, color='red')
    plt.savefig('South_Pacific_Ocean_corr.jpg', bbox_inches='tight')

def new_corr_c():
    fig = plt.figure(figsize=(8, 4))
    grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[0, 0])
    infos = cal_corr_info(mask_name='North_Atlantic_ocean.nc', corr='personr')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{c.}$', fontsize=28, x=-0.12, y=0.85, color='blue')
    plt.savefig('North_Atlantic_ocean_corr.jpg', bbox_inches='tight')

def new_corr_d():
    fig = plt.figure(figsize=(8, 4))
    grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
    # Equatorial_Indian_Ocean information
    ax2 = plt.subplot(grid[0, 0])
    infos = cal_corr_info(mask_name='california_west_coast.nc', corr='personr')
    ax2.bar(usecols, infos, color=colorlist, linewidth=2, **{'edgecolor': 'black'})
    ax2.set_xticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(r'$\bf{d.}$', fontsize=28, x=-0.12, y=0.85, color='blue')
    plt.savefig('california_west_coast_corr.jpg', bbox_inches='tight')

def draw_extend_fig_5():

    fig = plt.figure(figsize=(24, 12))
    grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.1)
    # ax = plt.subplot(grid[:3, :3], projection=ccrs.Mollweide(central_longitude=-155))
    # ax.coastlines()
    # ax.set_global()
    # ax.add_feature(cfeature.LAND, color='#CBCCCA')
    # ax.gridlines(linestyle='--', color='black', alpha = 0.5)

    dataset = xr.open_dataset('region_mask.nc')
    x_major_locator = MultipleLocator(3)
    y_major_locator = MultipleLocator(10)

    ax3 = plt.subplot(grid[0, 0])
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('Equatorial_Indian_Ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax3.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax3.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax3.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax3.set_xticks(np.arange(2003, 2020)[::4])
    ax3.legend(loc=2)
    ax3.grid()
    ax3.set_ylim(-30, 30)
    ax3.xaxis.set_major_locator(x_major_locator)
    ax3.yaxis.set_major_locator(y_major_locator)
    ax3.set_title(r'$\bf{a.}$', fontsize=28, x=-0.1, y=0.9, color='red')


    ax5 = plt.subplot(grid[0, 1], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('South_Pacific_Ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax5.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax5.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax5.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax5.set_xticks(np.arange(2003, 2020)[::4])
    ax5.legend(loc=2)
    ax5.grid()
    ax5.set_ylim(-30, 30)
    ax5.xaxis.set_major_locator(x_major_locator)
    ax5.yaxis.set_major_locator(y_major_locator)
    ax5.set_title(r'$\bf{b.}$', fontsize=28, x=-0.1, y=0.9, color='red')


    
    ax7 = plt.subplot(grid[1, 0], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('North_Atlantic_ocean.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax7.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax7.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax7.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax7.set_xticks(np.arange(2003, 2020)[::4])
    ax7.legend(loc=2)
    ax7.grid()
    ax7.set_ylim(-30, 30)
    ax7.xaxis.set_major_locator(x_major_locator)
    ax7.yaxis.set_major_locator(y_major_locator)
    ax7.set_title(r'$\bf{c.}$', fontsize=28, x=-0.1, y=0.9, color='blue')


    ax9 = plt.subplot(grid[1, 1], )
    temp_mask = dataset.where(xr.open_dataset('./mask/{}'.format('california_west_coast.nc')).__xarray_dataarray_variable__)
    depth_300, yeer_300, depth_1000, yeer_1000, depth_3000, yeer_3000 = get_region_do_concentration(temp_mask=temp_mask)
    ax9.errorbar(np.arange(2003, 2021), depth_300, yerr=yeer_300, label='300m', color='g', linewidth=2)
    ax9.errorbar(np.arange(2003, 2021)+0.1, depth_1000, yerr=yeer_1000, label='1046m', color='orange', linewidth=2)
    ax9.errorbar(np.arange(2003, 2021)+0.2, depth_3000, yerr=yeer_3000, label='2956m', color='blue', linewidth=2)
    ax9.set_xticks(np.arange(2003, 2020)[::4])
    ax9.legend(loc=2)
    ax9.grid()
    ax9.set_ylim(-30, 30)
    ax9.xaxis.set_major_locator(x_major_locator)
    ax9.yaxis.set_major_locator(y_major_locator)
    ax9.set_title(r'$\bf{d.}$', fontsize=28, x=-0.1, y=0.9, color='blue')


    plt.savefig('extended_figure_5.png', bbox_inches='tight')



if __name__ == '__main__':
    # draw_mutual_info_bar()
    new_draw_feature_trend_a()
    # new_draw_feature_trend_b()
    # new_draw_feature_trend_c()
    # new_draw_feature_trend_d()
    
    # draw_pearconr_info_bar()
    # draw_mutual_info_bar_do_content()
    # get_map()
    # draw_colorbar()
    # draw_head()
    # new_corr_a()
    # new_corr_b()
    # new_corr_c()
    # new_corr_d()
    # draw_extend_fig_5()
    
    exit()
    # do_tot = xr.open_dataset('./do_content_per_site_remap.nc')
    # mask_list = ['Equatorial_Indian_Ocean.nc', 'North_Atlantic_ocean.nc', 'South_Pacific_Ocean.nc', 'california_west_coast.nc']
    # year = np.arange(2003, 2021)
    # fig = plt.figure(figsize=(12, 4))

    # for mask_name in mask_list:
    #     mask = xr.open_dataset('./mask/{}'.format(mask_name))
    #     do_tot_mask = do_tot.where(mask.__xarray_dataarray_variable__).sum(dim=['lon', 'lat'])
    #     # print(mask_name, do_tot_mask.__xarray_dataarray_variable__.values / 10**12)
    #     do_tot_mask = do_tot_mask.__xarray_dataarray_variable__.values / 10**12
    #     # print(mask_name, (do_tot_mask[-1]-do_tot_mask[0]) / do_tot_mask[0])
    #     plt.plot(year, do_tot_mask, color='#EF7A6D', marker='o')
    #     for x, y in zip(year, do_tot_mask):
    #         plt.text(x+0.2, y + 0.1, '%.2f' % y, ha='center', va='bottom', fontsize=8)
        
    #     plt.title(mask_name[:-3])
    #     new_ticks = np.arange(2003, 2021)[::2]
    #     plt.xticks(new_ticks)
    #     plt.savefig(mask_name[:-3]+'.jpg', bbox_inches='tight')
    #     plt.clf()

    # import pdb; pdb.set_trace()