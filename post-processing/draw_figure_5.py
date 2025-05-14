import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import MultipleLocator


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams.update({'font.size': 14})

def my_func(a): 
    mask = a == 0
    if np.all(mask): return np.nan
    return LinearRegression().fit(np.arange(2003, 2021)[~mask].reshape(-1, 1), a[~mask]).coef_


def func_interval_bootstrap(a, iteration=500):
    mask = a == 0
    if np.all(mask): return np.nan, np.nan
    values = np.hstack([np.arange(2003, 2021)[~mask].reshape(-1, 1), a[~mask].reshape(-1, 1)])
    index = np.arange(0, values.shape[0])
    
    result_iter = []
    for k in range(iteration):
        sample = values[np.random.choice(index, len(index), replace=True)]
        result_iter.append(LinearRegression().fit(sample[:, 0].reshape(-1, 1), sample[:, 1]).coef_)
    
    result_iter = np.array(result_iter)
    return np.quantile(result_iter, 0.025), np.quantile(result_iter, 0.975)
    

def draw_figure2_a(path):
    data = xr.open_dataset(path)
    values = data.__xarray_dataarray_variable__.values[0]
    lats = data.lat.values
    result = {}
    for lat, val in zip(lats, values):
        mask = np.isnan(val)
        if np.all(mask): result[lat] = np.nan
        else:
            
            result[lat] = val[~mask].sum() / 10**15
    fig = plt.figure(figsize=(12, 4))
    val = result.values()
    lat = result.keys()
    plt.plot(lat, val)
    plt.ylim(0, 5)
    plt.grid()
    plt.ylabel('Latitudinal $O_2$ (Pmol per degree)')
    plt.fill_between(lat, val, 0, facecolor='grey')
    plt.savefig('figure2_a.jpg')
    
    # import pdb; pdb.set_trace()


def generate_change_do_nc():
    do_content_per_site_avg = xr.open_dataset('../do_content_per_site_avg.nc')
    trend = np.load('../trend_lr.npy') / 10**3 * 10
    data = xr.DataArray(data=trend, dims=['lat', 'lon'], coords=dict(lon=do_content_per_site_avg.lon.data, lat=do_content_per_site_avg.lat.data))
    data.to_netcdf('./cdo_temp.nc') 
    os.system('cdo griddes ./cdo_temp.nc > mygrid')
    os.system('sed -i "s/generic/lonlat/g" mygrid')
    os.system('cdo setgrid,mygrid cdo_temp.nc do_content_per_site_trend.nc')
    os.system('rm -rf mygrid cdo_temp.nc')
    os.system('cdo remapbil,r360x181 do_content_per_site_trend.nc do_content_per_site_trend_1d.nc')
    os.system('cdo -z zip9 -mul do_content_per_site_trend_1d.nc -gridarea do_content_per_site_trend_1d.nc do_content_per_site_trend_1d_cont.nc')
    

def draw_figure2_b_new():
    os.system('cdo remapbil,r360x181 ../do_content_per_site.nc do_content_per_site_1d.nc')
    os.system('cdo -z zip9 -mul do_content_per_site_1d.nc -gridarea do_content_per_site_1d.nc do_content_per_site_1d_cont.nc')

    data = xr.open_dataset('do_content_per_site_1d_cont.nc').sum(dim=['lon'])
    lats = data.lat.values
    
    result = []
    interval_lower_bound = []
    interval_upper_bound = []
    
    for val in lats:
        si_data = data.__xarray_dataarray_variable__.sel(lat=val) / 10**12
        trend = my_func(si_data.values) * 10
        result.append(trend) 
        lower_bound, upper_bound = func_interval_bootstrap(si_data.values) #* 10
        interval_lower_bound.append(lower_bound * 10)
        interval_upper_bound.append(upper_bound * 10) 
        
    fig = plt.figure(figsize=(12, 4))
    val = result
    lower_bound = interval_lower_bound
    upper_bound = interval_upper_bound
    plt.plot(lats, val)

    plt.fill_between(lats, lower_bound, upper_bound, facecolor='#FFD2D2', edgecolor='white', linewidth=0)
    plt.yticks([5, 0, -5, -10, -15, -20, -25, -30])

    plt.grid()
    plt.ylabel('Change in latitudinal $O_2$ (Tmol per degree per decade)', fontsize=12)
    plt.savefig('figure2_b.jpg')


def draw_figure2_c():
    data = xr.open_dataset('mask_2003-2020_yearly_mean_nochlmask.nc')
    o2 = data.o2
    if not os.path.exists('single_level_do_concentration'): os.makedirs('single_level_do_concentration')
    for i in range(o2.depth.shape[0]):
        single_level_data = o2.isel(depth=i)* 1026 / 10**6 # convert o2 ((umol/kg * rho (kg/m^3) / 10^6) to mol/m^3)
    
        single_level_data.to_netcdf('./cdo_temp.nc') 
        os.system('cdo griddes ./cdo_temp.nc > mygrid')
        os.system('sed -i "s/generic/lonlat/g" mygrid')
        os.system('cdo setgrid,mygrid cdo_temp.nc {}/doc_{}.nc'.format('single_level_do_concentration', str(i)))
        os.system('rm -rf mygrid cdo_temp.nc')
        os.system('cdo -z zip9 -fldsum -mul {}/doc_{}.nc -gridarea {}/doc_{}.nc {}/tot_doc_{}.nc'.format('single_level_do_concentration', str(i), 'single_level_do_concentration', str(i), 'single_level_do_concentration', str(i)))
    re, depth = [], []
    for i in range(o2.depth.shape[0]):
        da = xr.open_dataset('{}/tot_doc_{}.nc'.format('single_level_do_concentration', str(i))).o2
        depth.append(da.depth.values)
        re.append(da.values.squeeze() / 10**12) 
    
    fig = plt.figure(figsize=(6, 12), tight_layout=True)
    plt.plot(re, depth)
    plt.ylim(6000, 0)
    plt.xlim(0, 110)
    plt.yticks([0, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000])
    plt.xlabel('Vertical $O_2$ (Tmol $m^{-1}$)', fontsize=12)
    plt.grid()
    plt.fill_between(re, depth, 0, facecolor='grey')
    plt.savefig('figure2_c.jpg')


def draw_figure2_d_new(generate_flag=False):
    totdata = xr.open_dataset('../../experiments/mask_2003-2020_yearly.nc')
    if generate_flag:
        mask = np.load('../mask_2003-2020_yearly_mask.npy')
        o2 = totdata.o2.values * 1026 / 10**6 # convert o2 ((umol/kg * rho (kg/m^3) / 10^6) to mol/m^3)
    
        o2[:, :, ~mask] = np.nan
        if not os.path.exists('single_level_do_concentration_trend'): os.makedirs('single_level_do_concentration_trend')
        for index, depth in enumerate(totdata.depth):
            data = xr.DataArray(data=o2[:, index, :, :], dims=['time', 'lat', 'lon'], coords=dict(time=totdata.time.data, lon=totdata.lon.data, lat=totdata.lat.data))
            data.to_netcdf('./cdo_temp.nc') 
            os.system('cdo griddes ./cdo_temp.nc > mygrid')
            os.system('sed -i "s/generic/lonlat/g" mygrid')
            os.system('cdo setgrid,mygrid cdo_temp.nc single_level_do_concentration_trend/doc_{}.nc'.format(index))
            os.system('rm -rf mygrid cdo_temp.nc')
            os.system('cdo -z zip9 -fldsum -mul {}/doc_{}.nc -gridarea {}/doc_{}.nc {}/tot_doc_{}.nc'.format('single_level_do_concentration_trend', str(index), 'single_level_do_concentration_trend', str(index), 'single_level_do_concentration_trend', str(index)))
        
    re, depth = [], []
    interval_lower_bound = []
    interval_upper_bound = []
    for index, de in enumerate(totdata.depth.values):
        data = xr.open_dataset('{}/tot_doc_{}.nc'.format('single_level_do_concentration_trend', str(index))).__xarray_dataarray_variable__.values.squeeze() / 10**12
        trend = my_func(data) * 10
        depth.append(de)
        re.append(trend)
        lower_bound, upper_bound = func_interval_bootstrap(data)
        interval_lower_bound.append(lower_bound * 10)
        interval_upper_bound.append(upper_bound * 10)   
    
    fig = plt.figure(figsize=(6, 12), tight_layout=True)
    plt.plot(re, depth, color='black', linewidth=3)
    plt.fill_betweenx(depth, interval_lower_bound, interval_upper_bound, facecolor='#FFD2D2', edgecolor='white', linewidth=0)

    plt.ylim(6000, 0)
    plt.xlim(-0.7, 0.2)
    plt.yticks([0, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000])
    plt.xlabel('Change in vertical $O_2$ (Tmol $m^{-1}$ per decade)', fontsize=12)
    plt.grid()
    plt.savefig('figure2_d.jpg')


def draw_total_figure2(fontsize=15):
    fig = plt.figure(figsize=(24, 10))
    grid = plt.GridSpec(3, 7, wspace=0.35, hspace=0.4)

    ax1 = plt.subplot(grid[:2, :3], projection=ccrs.Mollweide(central_longitude=-155))
    data = xr.open_dataset('../do_content_per_site_avg.nc')
    lon = data.lon.values
    lat = data.lat.values
    value = data.__xarray_dataarray_variable__.values[0] / 10**3
    levels = np.linspace(0, 1.8, 10)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, color='#CBCCCA')
    p = ax1.contourf(lon, lat, value, transform=ccrs.PlateCarree(), cmap='YlOrBr', levels=levels)
    cb = plt.colorbar(p, pad=0.08, orientation='horizontal', shrink=0.8, label='$O_2$ content ($kmol \cdot m^{-2}$)') #cax=position, 
    ax1.gridlines(linestyle='--', color='black', alpha = 0.5)
    ax1.set_title(r'$\bf{a.}$', fontsize=28, x= 0.05, y=0.95)


    ax2 = plt.subplot(grid[:2, 3:6], projection=ccrs.Mollweide(central_longitude=-155))
    data = xr.open_dataset('../do_content_per_site.nc')
    lon = data.lon.values
    lat = data.lat.values
    data = data.__xarray_dataarray_variable__.values
    if not os.path.exists('../trend_lr.npy'): trend = np.apply_along_axis(my_func, 0, data); np.save('../trend_lr.npy', trend)
    else: trend = np.load('../trend_lr.npy')

    value = trend * 10
    levels = np.linspace(-40, 40, 17)
    value[value>40] = 40
    value[value<-40] = -40
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, color='#CBCCCA')
    colors = [ '#E10414', '#DC2828', '#DC4646', '#DC645A', '#E67D7D', '#E6A0A0', '#F0BEBE', '#F4DFDD',
            '#FFFFFF', '#FFFFFF', '#FFFFFF', '#DDDDEF', '#BEBEDC', '#9696C8', '#6E78B4', '#4A58A6', '#354FA0', '#354B9D', '#19499D']
    
    my_cmap = ListedColormap(colors, name="my_cmap")
    p = ax2.contourf(lon, lat, value, transform=ccrs.PlateCarree(), cmap=my_cmap, levels=levels)#'bwr_r'
    cb = plt.colorbar(p, pad=0.08, orientation='horizontal', shrink=0.8, label='Changes in $O_2$ ($mol \cdot m^{-2} \cdot decade$)') #cax=position, 
    ax2.gridlines(linestyle='--', color='black', alpha = 0.5)
    ax2.set_title(r'$\bf{b.}$', fontsize=28, x= 0.05, y=0.95)

    ax3 = plt.subplot(grid[:, 6], autoscale_on=True)
    data = xr.open_dataset('mask_2003-2020_yearly_mean_nochlmask.nc')
    o2 = data.o2
    re, depth = [], []
    for i in range(o2.depth.shape[0]):
        da = xr.open_dataset('{}/tot_doc_{}.nc'.format('single_level_do_concentration', str(i))).o2
        depth.append(da.depth.values)
        re.append(da.values.squeeze() / 10**12) 
    ax3.plot(re, depth)
    ax3.set_ylim(6000, 0)
    ax3.set_xlim(0, 80)
    
    y_major_locator=MultipleLocator(200)
    ax3.yaxis.set_major_locator(y_major_locator)
    ax3.set_yticks(np.arange(0, 6200, 200)[::2])
    ax3.set_xticks(np.linspace(0, 80, 5))
    ax3.set_xlabel('Vertical $O_2$ (Tmol $\cdot$ $m^{-1}$)', fontsize=fontsize)
    ax3.set_ylabel('Depth (m)')

    ax3.grid()
    ax3.fill_between(re, depth, 0, facecolor='#CBCCCA', edgecolor='#CBCCCA', alpha=1)
    ax3_twin = ax3.twiny()
    re, depth = [], []
    interval_lower_bound = []
    interval_upper_bound = []

    for index, de in enumerate(data.depth.values):
        tot = xr.open_dataset('{}/tot_doc_{}.nc'.format('single_level_do_concentration_trend', str(index))).__xarray_dataarray_variable__.values.squeeze() / 10**12
        trend = my_func(tot) * 10
        depth.append(de)
        re.append(trend)
        lower_bound, upper_bound = func_interval_bootstrap(tot)
        interval_lower_bound.append(lower_bound * 10)
        interval_upper_bound.append(upper_bound * 10)  
    
    ax3_twin.plot(re, depth, color='red', linewidth=3)
    ax3_twin.fill_betweenx(depth, interval_lower_bound, interval_upper_bound, facecolor='#FFD2D2', edgecolor='white', linewidth=0, alpha=0.5) #D4DFF8
    
    ax3_twin.set_ylim(6000, 0)
    ax3_twin.set_xlim(-0.6, 0.18)
    ax3_twin.set_xticks(np.arange(-0.6, 0.4, 0.2))
    ax3_twin.set_xlabel('Change in vertical $O_2$\n(Tmol $\cdot$ $m^{-1}$ $\cdot$ decade)', fontsize=fontsize)
    ax3_twin.grid()
    ax3_twin.xaxis.label.set_color('red')
    ax3_twin.tick_params(axis='x', colors='red')
    ax3_twin.set_title(r'$\bf{c.}$', fontsize=28, x= -0.4, y=0.99)


    ax4 = plt.subplot(grid[2, 1:5], autoscale_on=True)
    data = xr.open_dataset('./do_content_per_site_avg_1d_cont.nc')
    values = data.__xarray_dataarray_variable__.values[0]
    lats = data.lat.values
    result = {}
    for lat, val in zip(lats, values):
        mask = np.isnan(val)
        if np.all(mask): result[lat] = np.nan
        else:   
            result[lat] = val[~mask].sum() / 10**15
    val = np.array(list(result.values()))
    lat = np.array(list(result.keys()))
    
    ax4.plot(lat, val)

    ax4.set_ylim(0, 5)
    ax4.grid()
    ax4.set_ylabel('Latitudinal $O_2$ \n(Pmol $\cdot$ degree)', fontsize=fontsize)
    ax4.fill_between(lat, val, 0, facecolor='#CBCCCA', edgecolor='white', linewidth=0)
    ax4.set_yticks(np.arange(0, 6))
    
    ax4_twin = ax4.twinx()
    data = xr.open_dataset('do_content_per_site_1d_cont.nc').sum(dim=['lon'])
    lats = data.lat.values
    result = []
    interval_lower_bound = []
    interval_upper_bound = []
    for val in lats:
        si_data = data.__xarray_dataarray_variable__.sel(lat=val) / 10**12
        trend = my_func(si_data.values) * 10
        if isinstance(trend, np.ndarray):
            trend = trend[0]
        result.append(trend) 
        lower_bound, upper_bound = func_interval_bootstrap(si_data.values) #* 10
        interval_lower_bound.append(lower_bound * 10)
        interval_upper_bound.append(upper_bound * 10) 
    val = np.array(result)
    lower_bound = interval_lower_bound
    upper_bound = interval_upper_bound

    ax4_twin.plot(lat, val, color='red', linewidth=3)
    ax4_twin.fill_between(lats, lower_bound, upper_bound, facecolor='#FFD2D2', edgecolor='white', linewidth=0, alpha=0.5)
    
    ax4_twin.set_ylabel('Change in latitudinal $O_2$ \n(Tmol $\cdot$ degree $\cdot$ decade)', fontsize=fontsize)
    ax4_twin.grid() #linestyle='--'
    ax4_twin.set_yticks([-48, -34, -20, -6, 8, 22])
    ax4_twin.set_yticklabels(['-48', '-34', '-20', ' -6 ','  8', ' 22'])
    ax4_twin.yaxis.label.set_color('red')
    ax4_twin.tick_params(axis='y', colors='red')
    ax4_twin.set_title(r'$\bf{d.}$', fontsize=28, x= -0.08, y=0.95)

    ax4_twin.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    ax4_twin.set_xticklabels(['75$^{\circ}$S', '50$^{\circ}$S', '25$^{\circ}$S', '0$^{\circ}$','25$^{\circ}$N', '50$^{\circ}$N', '75$^{\circ}$N'])
    plt.savefig('figure_3.png', bbox_inches='tight')#, 

    
if __name__ == '__main__':
    
    draw_total_figure2()
