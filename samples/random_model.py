# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
basically to show how to use NNGenerater1d
sample code from https://morvanzhou.github.io/tutorials/
"""

import sys

sys.path.append('../')
from XenonPy import ElementDesc, Just, load
from XenonPy import Path
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from XenonPy import NNGenerator1d, ModelRunner
import seaborn as sb
import torch

from matplotlib import pyplot as plt


def po_plot(y_pred, y_true, fname: str = None, describe: str = None):
    fname_ext = fname.split('.')
    name, ext = fname_ext[0], fname_ext[1]
    import matplotlib as mpl
    with mpl.rc_context(rc={'font.size': 25}):
        ax = sb.jointplot(y_pred, y_true, kind="reg", size=10)
        ax.set_axis_labels('Prediction', 'Observation')
        if describe:
            ax.fig.subplots_adjust(top=0.9)
            ax.fig.suptitle(describe)
        ax.savefig(name + '_reg.' + ext, dpi=150, bbox_inches='tight')

        ax = sb.jointplot(y_pred, y_true, kind="kde", size=10)
        ax.set_axis_labels('Prediction', 'Observation')
        if describe:
            ax.fig.subplots_adjust(top=0.9)
            ax.fig.suptitle(describe)
        ax.savefig(name + '_kde.' + ext, dpi=150, bbox_inches='tight')


def cb_plot(y, y_cb, fname: str = None, describe: str = None):
    fname_ext = fname.split('.')
    name, ext = fname_ext[0], fname_ext[1]
    import matplotlib as mpl
    with mpl.rc_context(rc={'font.size': 25, 'figure.figsize': (20, 10)}):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sb.distplot(y, ax=ax1)
        sb.distplot(y_cb, ax=ax2)
        plt.title(describe)
        plt.savefig(fname)


# torch.manual_seed(1)    # reproducible
print('loading data...')
ox = load('mp_inorganic')
ox.info()

print('')
# ox = ox[(ox.e_above_hull == 0) & (ox.band_gap != 0) & ox.has_bandstructure]
ox = ox[ox.e_above_hull == 0]

print('calculate descriptor...')
desc = ElementDesc()
desc = Just(ox) >> desc
desc = ~desc
desc.info()

ox_vol = ox.reset_index(drop=True)
ox_vol = ox_vol[ox_vol.volume != 0]
ox_vol = ox_vol.sort_values(by='volume', ascending=False)['volume']
desc_vol = desc.iloc[ox_vol.index, :]

# %% box-cox
ox_vol_bc, lmd = boxcox(ox_vol - ox_vol.min() + 1E-6)
scaler = StandardScaler()
desc_vol_scale = scaler.fit_transform(desc_vol)

# %% net
n_neuron = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
generator = NNGenerator1d(290, 1, n_neuron=n_neuron, p_drop=np.arange(0.02, 0.5, 0.04),
                          batch_normalize=[True], momentum=np.arange(0.1, 0.6, 0.1))
X_train, X_test, y_train, y_test = train_test_split(desc_vol_scale, ox_vol_bc, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

model_num = 10000
i = 0
while i < model_num:
    model = generator(2)
    runner = ModelRunner(model, ctx='GPU', lr=0.4, loss_func=torch.nn.MSELoss(), optim=torch.optim.SGD, verbose=200,
                         checkstep=200,
                         epochs=2500, save_to='D:\\nn-net\\net_' + str(i))
    runner(X_train, y_train)
    y_true, y_pred = y_test, runner.predict(
        torch.from_numpy(X_test).type(torch.cuda.FloatTensor)).cpu().data.numpy().flatten()
    _p = str(Path('D:\\nn-net\\net_' + str(i)) / 'ox_vol_bc_net.png')
    po_plot(y_pred, y_true, _p, describe='NN (box-cox)')
    i += 1

# plt.ion()  # something about plotting

# for t in range(200):
#     prediction = net(x)  # input x and predict based on x
#
#     loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
#
#     optimizer.zero_grad()  # clear gradients for next train
#     loss.backward()  # backpropagation, compute gradients
#     optimizer.step()  # apply gradients
#
#     if t % 5 == 0:
#         # plot and show learning process
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
