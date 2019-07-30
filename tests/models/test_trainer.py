#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pytest
import torch
import torch.nn.functional as F

from xenonpy.model.training import Trainer, MSELoss, Adam


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

        def forward(self, x_):
            x_ = F.relu(self.hidden(x_))  # activation function for hidden layer
            x_ = self.predict(x_)  # linear output
            return x_

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    net = Net(n_feature=1, n_hidden=10, n_output=1)

    yield net, (x, y)

    print('test over')


def test_trainer_1(data):
    trainer = Trainer(optimizer=Adam(), loss_func=MSELoss())
    assert trainer.device == torch.device('cpu')
    assert trainer.model is None
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer.x_val is None
    assert trainer.y_val is None
    assert trainer.validate_dataset is None
    assert trainer._init_states is None
    assert trainer._init_optim is None
    assert trainer._scheduler is None
    assert isinstance(trainer._optim, Adam)
    assert isinstance(trainer.loss_func, MSELoss)

    trainer.model = data[0]
    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer._init_optim, dict)
    assert isinstance(trainer._init_states, dict)


def test_trainer_2(data):
    model = data[0]
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss())
    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer._init_optim, dict)
    assert isinstance(trainer._init_states, dict)


if __name__ == "__main__":
    pytest.main()
