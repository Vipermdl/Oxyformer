import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams.update({'font.size': 18})


fig = plt.figure(figsize=(24, 12))
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.1)

cmap = 'tab20c'


colors = [
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
        '#cab2d6',
    ]

colors = [
'#fff7ec',
'#fee8c8',
'#fdd49e',
'#fdbb84',
'#fc8d59',
'#ef6548',
'#d7301f',
'#b30000',
'#7f0000',
]
cmap = ListedColormap(colors[::-1], name="my_cmap")



hot = xr.open_dataset('./bats_oxyformer.nc')
time = hot.time.data
depth = hot.depth.data
levels = np.linspace(130, 280, 20)
# depth less than 300m
ax1 = plt.subplot(grid[0, 0])
value = hot.sel(depth=depth).o2.data.T
p = ax1.contourf(time, depth, value, cmap=cmap, levels=levels)
ax1.set_ylim(2000, 0)
ax1.set_yticks(np.arange(0, 2100, 200))
ax1.set_ylabel('Depth')
ax1.set_title(r'$\bf{a.}$', fontsize=28, x= -0.1, y=0.95)
# ax2.set_xticks([])


truth = xr.open_dataset('./bats_truth.nc')
truth = truth.isel(time=np.arange(0, 215)[:-16])
time = truth.sel(lat=31.75, lon=64.25).time.data
depth = truth.sel(lat=31.75, lon=64.25).depth.data
ax2 = plt.subplot(grid[1, 0])
value = truth.sel(lat=31.75, lon=64.25, depth=
                  depth).o2.data.T

import pdb; pdb.set_trace()

p = ax2.contourf(time, depth, value, cmap=cmap, levels=levels) # vmin=130, vmax=280
ax2.set_ylim(2000, 0)
ax2.set_yticks(np.arange(0, 2100, 200))
ax2.set_ylabel('Depth')
ax2.set_title(r'$\bf{b.}$', fontsize=28, x= -0.1, y=0.95)

position = fig.add_axes([0.125, 0.03, 0.352, .04 ]) #[0.92, 0.11, 0.015, .77 ]
cb = plt.colorbar(p, cax=position, orientation='horizontal')
colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
cb.ax.tick_params(labelsize=18, direction='in')
# plt.title('Dissolved Oxygen ($\mu mol \cdot kg^{-1}$)', x=-10)


levels = np.linspace(20, 250, 20)
hot = xr.open_dataset('./hot_oxyformer.nc')
time = hot.time.data
depth = hot.depth.data

ax3 = plt.subplot(grid[0, 1])
value = hot.sel(depth=depth).o2.data.T
p = ax3.contourf(time, depth, value, cmap=cmap, levels=levels)
ax3.set_ylim(2000, 0)
ax3.set_yticks(np.arange(0, 2100, 200))
ax3.set_ylabel('Depth')
ax3.set_title(r'$\bf{e.}$', fontsize=28, x= -0.1, y=0.95)

# ax2.set_xticks([])
truth = xr.open_dataset('./hot_truth.nc')
truth = truth.isel(time=np.arange(0, 181)[:-9])
time = truth.sel(lat=22.75, lon=-158).time.data
depth = truth.sel(lat=22.75, lon=-158).depth.data

ax4 = plt.subplot(grid[1, 1])
value = truth.sel(lat=22.75, lon=-158, depth=depth).o2.data.T
p = ax4.contourf(time, depth, value, cmap=cmap, levels=levels) # vmin=130, vmax=280
ax4.set_ylim(2000, 0)
ax4.set_yticks(np.arange(0, 2100, 200))
ax4.set_ylabel('Depth')
ax4.set_title(r'$\bf{f.}$', fontsize=28, x= -0.1, y=0.95)


position = fig.add_axes([0.55, 0.03, 0.35, .04 ]) #[0.92, 0.11, 0.015, .77 ]
cb = plt.colorbar(p, cax=position, orientation='horizontal')
colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
cb.ax.tick_params(labelsize=18, direction='in')
# plt.title('Dissolved Oxygen ($\mu mol \cdot kg^{-1}$)', x=-10)
plt.savefig('hot_timeseries_v2.png', bbox_inches='tight', dpi=300)