"""
basically to show how to use Layer class
sample code from https://morvanzhou.github.io/tutorials/
"""
import sys

sys.path.append('/Users/liuchang/projects/python/XenonPy')
from XenonPy.model.nn import Layer1d
from torch import nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(
    torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(
    x.size())  # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


def Net(n_feature, n_hidden, n_output):
    return nn.Sequential(
        Layer1d(n_in=n_feature, n_out=n_hidden),  # hidden layer
        Layer1d(n_in=n_hidden, n_out=n_output, act_func=None),  # output layer
    )


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()  # something about plotting

for t in range(200):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(
            0.5,
            0,
            'Loss=%.4f' % loss.data[0],
            fontdict={
                'size': 20,
                'color': 'red'
            })
        plt.pause(0.1)

plt.ioff()
plt.show()
