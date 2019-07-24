#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pytest
import torch
import torch.nn.functional as F

from xenonpy.model.training.base import BaseRunner


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            return x

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    net = Net(n_feature=1, n_hidden=10, n_output=1)

    yield net, (x, y)

    print('test over')


def test_base_runner_1(data):
    torch.manual_seed(1)  # reproducible
    assert BaseRunner.check_cuda(False).type == 'cpu'
    assert BaseRunner.check_cuda('cpu').type == 'cpu'

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda(True)

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda('cuda')

    with pytest.raises(RuntimeError, match='wrong device identifier'):
        BaseRunner.check_cuda('other illegal')


if __name__ == "__main__":
    pytest.main()
