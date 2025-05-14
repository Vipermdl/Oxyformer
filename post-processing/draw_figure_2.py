import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
sys.path.append('../')
from lib.utils import labelcols

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = ['STSong', 'Times New Roman']

# plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams.update({'font.size': 14})

x_major_locator = MultipleLocator(100)
y_major_locator = MultipleLocator(100)

label = pd.read_csv('../dataset/data/test.csv', usecols=labelcols).values
oxyformer = pd.read_csv('./prediction_testset_outlier_check.csv', usecols=labelcols).values

mask = (~np.isnan(label)) & (~np.isnan(oxyformer)) & (np.isfinite(label))
label_mask = label[mask]
oxyformer_mask = oxyformer[mask]

x = label_mask.ravel() 
y = oxyformer_mask.ravel()

info = {
    'oxyformer': {'R':0.890, 'RMSE': 23.930, 'MAE': 15.930},
}

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(6, 10, wspace=0.0, hspace=0.0)

H, xedges, yedges = np.histogram2d(y, x, bins=150)
H = np.rot90(H)
H = np.flipud(H)
Hmasked = np.ma.masked_where(H==0,H)
norm = matplotlib.colors.Normalize(vmin=0, vmax=400)
ax = plt.subplot(grid[1:, :5])
im = ax.pcolormesh(xedges, yedges, Hmasked, cmap=cm.get_cmap('Spectral_r'), norm=norm) #cm.get_cmap('Spectral_r')
ax.text(0.74, 0.26, r'$R^2='+str(info['oxyformer']['R'])+'$', verticalalignment='top', transform=ax.transAxes, fontsize=10)
ax.text(0.74, 0.18, r'$MAE='+str(info['oxyformer']['MAE'])+'$', verticalalignment='top', transform=ax.transAxes, fontsize=10)
ax.text(0.74, 0.10, r'$RMSE='+str(info['oxyformer']['RMSE'])+'$', verticalalignment='top', transform=ax.transAxes, fontsize=10)
ax.set_xlabel('Oxyformer estimation ($\mu mol \cdot kg^{-1}$)')
ax.set_ylabel('Observational DO ($\mu mol \cdot kg^{-1}$)')
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_ticks([0,100,200,300,400])
ax.yaxis.set_ticks([0,100,200,300,400])
ax.set_xlim(0, 400)
ax.set_ylim(0, 400)
fig.subplots_adjust(right=0.9)
position = fig.add_axes([0.6, 0.11, 0.015, .64 ])
cb = fig.colorbar(im, cax=position)

ax_right = plt.subplot(grid[1:, 5:6])
# ax_right = sns.kdeplot(x, color='Red', shade = True, y=True)
ax_right.hist(list(x),200,facecolor='#525EA8',alpha=0.9, orientation='horizontal')
ax_right.yaxis.set_ticks([0,100,200,300,400])
ax_right.set_ylim(0, 400)
# ax_right.set_ylabel("Frequency")

ax_bottom = plt.subplot(grid[:1, :5])
ax_bottom.hist(list(y),200,facecolor='#525EA8',alpha=0.9, orientation='vertical')
ax_bottom.xaxis.set_ticks([0,100,200,300,400])
ax_bottom.set_xlim(0, 400)
ax_bottom.set_title(r'$\bf{a.}$', fontsize=18, x=-0.15, y=0.75, color='black')

# ax_bottom.invert_yaxis()

for item in [ax_bottom, ax_right]:
    item.set_xticks([])
    item.set_yticks([])
    item.spines['right'].set_visible(False)
    item.spines['top'].set_visible(False)
    item.spines['left'].set_visible(False)
    item.spines['bottom'].set_visible(False)
    


ax1 = plt.subplot(grid[:, 7:])
residual = label - oxyformer
res_value = residual[~np.isnan(label)]
ax1.hist(list(res_value), 200, facecolor='#525EA8', alpha=0.9, linewidth=1,)
s = pd.Series(res_value)
ax1.text(1.38, 0.26, r'$skewness='+str(round(s.skew(), 3))+'$', verticalalignment='top', transform=ax.transAxes, fontsize=10)
ax1.text(1.38, 0.18, r'$kurtosis='+str(round(s.kurt(), 3))+'$', verticalalignment='top', transform=ax.transAxes, fontsize=10)
ax1.set_xlabel('Residual ($\mu mol \cdot kg^{-1}$)')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1.set_title(r'$\bf{b.}$', fontsize=18, x=0.17, y=0.95, color='black')
ax1.set_ylabel('Frequency')
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.savefig(r'./scatter_One_Colorbar.png',dpi=900,bbox_inches='tight')


