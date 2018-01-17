# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#%%
# %matplotlib inline
import sys
sys.path.append('../')
import seaborn as sb
from XenonPy import ElementDesc, Just, load
from XenonPy import DescHeatmap

#%%
print('loading data...')
ox = load('mp_inorganic')
ox.info()

print('')
# ox = ox[(ox.e_above_hull == 0) & (ox.band_gap != 0) & ox.has_bandstructure]
ox = ox[ox.e_above_hull == 0]
ox.info()
# Index: 7805 entries, O2 to Na2Ca4ZrNbSi4O17F
print('')

#%%
print('calculate descriptor...')
desc = ElementDesc()
desc = Just(ox) >> desc
desc = ~desc
print(desc.info())

# 290 entries, ave:atomic_number to min:Polarizability
print('')

#%%
%matplotlib inline
print('draw heatmap\n')
ox_fm_en = ox[ox['formation_energy_per_atom'] != 0]
ox_fm_en = ox_fm_en.sort_values(
    by='formation_energy_per_atom',
    ascending=False)['formation_energy_per_atom']
desc_fm_en = desc.loc[ox_fm_en.index, :]

dh_map = DescHeatmap(
    save=dict(fname='elements_desc_fm_en.png', dpi=150, bbox_inches='tight', transparent=False),
    method='complete',
    figsize=(70, 10))

dh_map.fit(desc_fm_en)
dh_map.draw(ox_fm_en)

#%%
ox_bg = ox.reset_index(drop=True)
ox_bg = ox_bg[ox_bg['band_gap'] != 0]
ox_bg = ox_bg.sort_values(by='band_gap', ascending=False)['band_gap']
desc_bg= desc.iloc[ox_bg.index, :]
print(ox_bg.tail())

dh_map = DescHeatmap(
    save=dict(fname='elements_desc_bg.png', dpi=150, bbox_inches='tight', transparent=False),
    method='complete',
    figsize=(70, 10))

Just(desc_bg) >> dh_map << Just(ox_bg)

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def po_plot(y_pred, y_true, fname: str=None, describe: str=None):
    fname_ext = fname.split('.')
    name, ext = fname_ext[0], fname_ext[1]
    import matplotlib as mpl
    with mpl.rc_context(rc={'font.size': 25}):
        ax = sb.jointplot(y_pred, y_true, kind="reg", size=10)
        ax.set_axis_labels('Prediction', 'Observation')
        if describe:
            ax.fig.subplots_adjust(top=0.9)
            ax.fig.suptitle(describe)
        ax.savefig(name + '_reg.' +  ext , dpi=150, bbox_inches='tight')

        ax = sb.jointplot(y_pred, y_true, kind="kde", size=10)
        ax.set_axis_labels('Prediction', 'Observation')
        if describe:
            ax.fig.subplots_adjust(top=0.9)
            ax.fig.suptitle(describe)
        ax.savefig(name + '_kde.' +  ext , dpi=150, bbox_inches='tight')

def cb_plot(y, y_cb, fname: str=None, describe: str=None):
    fname_ext = fname.split('.')
    name, ext = fname_ext[0], fname_ext[1]
    import matplotlib as mpl
    with mpl.rc_context(rc={'font.size': 25, 'figure.figsize':(20, 10)}):
        fig, (ax1, ax2) = plt.subplots(1,2)
        sb.distplot(y, ax=ax1)
        sb.distplot(y_cb, ax=ax2)
        plt.title(describe)
        plt.savefig(fname)

#%%
scaler = StandardScaler()
desc_fm_scale = scaler.fit_transform(desc_fm_en)

X_train, X_test, y_train, y_test = train_test_split(desc_fm_scale, ox_fm_en, test_size=0.2, random_state=42)
kneigh = KNeighborsRegressor(weights='distance')
kneigh.fit(X_train, y_train)
y_true, y_pred = y_test, kneigh.predict(X_test)
po_plot(y_pred, y_true, 'ox_fm_en_kn.png',describe='KNeighbors')


#%%
from scipy.stats import boxcox
ox_fm_en_bc, lmd = boxcox(ox_fm_en-ox_fm_en.min() + 1E-6)
cb_plot(ox_fm_en, ox_fm_en_bc, fname='ox_fm_en_bc.png', describe='lamda: {:.5f}'.format(lmd))

#%%
X_train, X_test, y_train, y_test = train_test_split(desc_fm_scale, ox_fm_en_bc, test_size=0.2, random_state=42)
kneigh = KNeighborsRegressor(weights='distance', n_neighbors=10)
kneigh.fit(X_train, y_train)
y_true, y_pred = y_test, kneigh.predict(X_test)
po_plot(y_pred, y_true, 'ox_fm_en_bc_kn.png',describe='KNeighbors (box-cox)')


#%% random forest
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, random_state=42)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'ox_fm_en_bc_rf.png',describe='RandomForest (box-cox)')

#%% band gap
scaler = StandardScaler()
desc_bg_scale = scaler.fit_transform(desc_bg)
ox_bg_bc, lmd = boxcox(ox_bg - ox_bg.min() + 1e-6)
cb_plot(ox_bg, ox_bg_bc, fname='ox_bg_bc.png', describe='lamda: {:.5f}'.format(lmd))


#%% kneigh
X_train, X_test, y_train, y_test = train_test_split(desc_bg_scale, ox_bg_bc, test_size=0.2, random_state=42)
kneigh = KNeighborsRegressor(weights='distance', n_neighbors=10)
kneigh.fit(X_train, y_train)
y_true, y_pred = y_test, kneigh.predict(X_test)
po_plot(y_pred, y_true, 'ox_bg_bc_kn.png',describe='KNeighbors (box-cox)')

#%% random forest
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(desc_bg_scale, ox_bg_bc, test_size=0.2, random_state=42)
regr = RandomForestRegressor(max_depth=10, random_state=42)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'ox_bg_bc_rf.png',describe='RandomForest (box-cox)')

#%%
import sys
sys.path.append('../')
from pymatgen.ext.matproj import MPRester
from XenonPy.utils import PairDistributionFunction
mpr = MPRester('Zrp32nS1LVBHsGCK')
structures = mpr.get_structures("Fe")
species = ['Fe']
obj = PairDistributionFunction(structures[0], ngrid=101, rmax=15.0, cellrange=1, sigma=0.1, species=species, )
plt = obj.get_rdf_plot(ylim=(-0.005, 5.0),xlim=(0.5, 10.5))
plt.savefig('Al_Y-ref.png')

#%%
print('loading data...')
sample_A = load('sample_A')
sample_A.info()

#%%
print('calculate descriptor...')
desc = ElementDesc()
desc = ~(Just(sample_A) >> ElementDesc())
print(desc.head())

#%%
sample_A = sample_A.sort_values(by='props', ascending=True)['props']
desc_sample_A = desc.loc[sample_A.index, :]

#%%
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
desc_sa_scale = StandardScaler().fit_transform(desc_sample_A)
sample_A_bc, lmd = boxcox(sample_A)
cb_plot(sample_A, sample_A_bc, fname='sample_A_bc.png', describe='lamda: {:.5f}'.format(lmd))
X_train, X_test, y_train, y_test = train_test_split(desc_sa_scale, sample_A_bc, test_size=0.2, random_state=42)


#%%
from sklearn.neighbors import KNeighborsRegressor
kneigh = KNeighborsRegressor(weights='distance')
kneigh.fit(X_train, y_train)
y_true, y_pred = y_test, kneigh.predict(X_test)
po_plot(y_pred, y_true, 'sample_A_bc_kn.png',describe='KNeighbors (box-cox)')


#%%
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, n_jobs=-1)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'sample_A_bc_rf.png',describe='RandomForest (box-cox)')


#%%
from sklearn.linear_model import ElasticNetCV
regr = ElasticNetCV(cv=5, random_state=0)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'sample_A_bc_encv.png',describe='ElasticNetCV (box-cox)')


#%% ###########################
# volume
###########################
#%%
ox_vol = ox.reset_index(drop=True)
ox_vol = ox_vol[ox_vol.volume != 0]
ox_vol = ox_vol.sort_values(by='volume', ascending=False)['volume']
desc_vol= desc.iloc[ox_vol.index, :]

dh_map = DescHeatmap(
    save=dict(fname='elements_desc_vol.png', dpi=150, bbox_inches='tight', transparent=False),
    method='complete',
    figsize=(70, 10))

Just(desc_vol) >> dh_map << Just(ox_vol)

#%%
import torch
import torch.nn as nn
from XenonPy import BaseNet
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#%% box-cox
ox_vol_bc, lmd = boxcox(ox_vol-ox_vol.min() + 1E-6)
scaler = StandardScaler()
desc_vol_scale = scaler.fit_transform(desc_vol)

#%% random forest
X_train, X_test, y_train, y_test = train_test_split(desc_vol_scale, ox_vol_bc, test_size=0.2, random_state=42)
regr = RandomForestRegressor(max_depth=10, random_state=42, n_jobs=-1, verbose=1)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'ox_vol_bc_rf.png',describe='RandomForest (box-cox)')

#%% Gradient boosting
X_train, X_test, y_train, y_test = train_test_split(desc_vol_scale, ox_vol_bc, test_size=0.2, random_state=42)
regr = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01, loss='ls', verbose=1)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(X_test)
po_plot(y_pred, y_true, 'ox_vol_bc_gb.png',describe='GradientBoosting (box-cox)')

#%% net
X_train, X_test, y_train, y_test = train_test_split(desc_vol_scale, ox_vol_bc, test_size=0.2, random_state=42)
regr = BaseNet(ctx='GPU', loss_func=nn.MSELoss(), optim=torch.optim.Adam, verbose=100, lr=0.01, epochs=5000)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
regr.fit(X_train, y_train)
y_true, y_pred = y_test, regr.predict(torch.from_numpy(X_test).type(torch.cuda.FloatTensor)).cpu().data.numpy().flatten()
po_plot(y_pred, y_true, 'ox_vol_bc_net.png',describe='NN (box-cox)')