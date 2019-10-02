#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from xenonpy.model.training import Trainer, MSELoss, Adam, ExponentialLR, SGD, ClipValue, ReduceLROnPlateau, ClipNorm
from xenonpy.model.training.extension import TensorConverter, Persist


class _Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x_):
        x_ = F.relu(self.hidden(x_))  # activation function for hidden layer
        x_ = self.predict(x_)  # linear output
        return x_


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    model_dir: Path = Path(__file__).parent / 'model_dir'

    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    net = _Net(n_feature=1, n_hidden=10, n_output=1)

    yield net, (x, y)

    if model_dir.exists():
        shutil.rmtree(str(model_dir.resolve()))

    print('test over')


def test_trainer_1(data):
    trainer = Trainer()
    assert trainer.device == torch.device('cpu')
    assert trainer.model is None
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer.x_val is None
    assert trainer.y_val is None
    assert trainer.validate_dataset is None
    assert trainer._init_states is None
    assert trainer._optimizer_state is None
    assert trainer.total_epochs == 0
    assert trainer.total_iterations == 0
    assert trainer.training_info is None
    assert trainer.loss_type is None
    assert trainer.loss_func is None

    trainer = Trainer(optimizer=Adam(), loss_func=MSELoss(), lr_scheduler=ExponentialLR(gamma=0.99),
                      clip_grad=ClipValue(clip_value=0.1))
    assert isinstance(trainer._scheduler, ExponentialLR)
    assert isinstance(trainer._optim, Adam)
    assert isinstance(trainer.clip_grad, ClipValue)
    assert isinstance(trainer.loss_func, MSELoss)


def test_trainer_2(data):
    trainer = Trainer()
    with pytest.raises(RuntimeError, match='no model for training'):
        trainer.fit(*data[1])

    with pytest.raises(TypeError, match='parameter `m` must be a instance of <torch.nn.modules>'):
        trainer.model = {}

    trainer.model = data[0]
    assert isinstance(trainer.model, torch.nn.Module)
    with pytest.raises(RuntimeError, match='no loss function for training'):
        trainer.fit(*data[1])

    trainer.loss_func = MSELoss()
    assert trainer.loss_type == 'train_mse_loss'
    assert trainer.loss_func.__class__ == MSELoss
    with pytest.raises(RuntimeError, match='no optimizer for training'):
        trainer.fit(*data[1])

    trainer.optimizer = Adam()
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer._optimizer_state, dict)
    assert isinstance(trainer._init_states, dict)

    trainer.lr_scheduler = ExponentialLR(gamma=0.99)
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)


def test_trainer_3(data):
    model = data[0]
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss())
    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer._optimizer_state, dict)
    assert isinstance(trainer._init_states, dict)
    assert trainer.clip_grad is None
    assert trainer.lr_scheduler is None

    trainer.lr_scheduler = ExponentialLR(gamma=0.1)
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    trainer.optimizer = SGD()
    assert isinstance(trainer.optimizer, torch.optim.SGD)

    trainer.clip_grad = ClipNorm(max_norm=0.4)
    assert isinstance(trainer.clip_grad, ClipNorm)


def test_trainer_fit_1(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss())
    trainer.fit(*data[1])
    assert trainer.total_iterations == 200
    assert trainer.total_epochs == 200

    trainer.fit(*data[1], epochs=20)
    assert trainer.total_iterations == 220
    assert trainer.total_epochs == 220

    trainer.reset()
    assert trainer.total_iterations == 0
    assert trainer.total_epochs == 0

    trainer.fit(*data[1], epochs=20)
    assert trainer.total_iterations == 20
    assert trainer.total_epochs == 20

    assert isinstance(trainer.training_info, pd.DataFrame)
    assert 'i_epoch' in trainer.training_info.columns

    ret = trainer.to_namedtuple()
    assert isinstance(ret, trainer.results_tuple)

    train_set = DataLoader(TensorDataset(*data[1]))
    with pytest.raises(RuntimeError, match='parameter <training_dataset> is exclusive of <x_train> and <y_train>'):
        trainer.fit(*data[1], training_dataset=train_set)

    with pytest.raises(RuntimeError, match='missing parameter <x_train> or <y_train>'):
        trainer.fit(data[1][0])


def test_trainer_fit_2(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss(), epochs=20)
    trainer.fit(*data[1], *data[1])
    assert trainer.total_iterations == 20
    assert trainer.total_epochs == 20
    assert (trainer.x_val, trainer.y_val) == data[1]

    train_set = DataLoader(TensorDataset(*data[1]))
    val_set = DataLoader(TensorDataset(*data[1]))
    trainer.fit(training_dataset=train_set, validation_dataset=val_set)
    assert trainer.total_iterations == 2020
    assert trainer.total_epochs == 40
    assert isinstance(trainer.validate_dataset, DataLoader)


def test_trainer_fit_3(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss(), epochs=5)
    trainer.fit(*data[1])
    assert len(trainer.checkpoints.keys()) == 0

    trainer.reset()
    assert trainer.total_iterations == 0
    assert trainer.total_epochs == 0
    assert len(trainer.get_checkpoint()) == 0

    trainer.fit(*data[1], checkpoint=True)
    assert len(trainer.get_checkpoint()) == 5
    assert isinstance(trainer.get_checkpoint(2), trainer.checkpoint_tuple)
    assert isinstance(trainer.get_checkpoint('cp_2'), trainer.checkpoint_tuple)

    with pytest.raises(TypeError, match='parameter <cp> must be str or int'):
        trainer.get_checkpoint([])

    trainer.reset(to=3, remove_checkpoints=False)
    assert len(trainer.get_checkpoint()) == 5
    assert isinstance(trainer.get_checkpoint(2), trainer.checkpoint_tuple)
    assert isinstance(trainer.get_checkpoint('cp_2'), trainer.checkpoint_tuple)

    trainer.reset(to='cp_3')
    assert trainer.total_iterations == 0
    assert trainer.total_epochs == 0
    assert len(trainer.get_checkpoint()) == 0

    with pytest.raises(TypeError, match='parameter <to> must be torch.nnModule, int, or str'):
        trainer.reset(to=[])

    # todo: need a real testing
    trainer.fit(*data[1], checkpoint=True)
    trainer.predict(*data[1], checkpoint=3)

    trainer.reset()
    trainer.fit(*data[1], checkpoint=lambda i: (True, f'new:{i}'))
    assert len(trainer.get_checkpoint()) == 5
    assert trainer.get_checkpoint() == list([f'new:{i + 1}' for i in range(5)])


def test_trainer_fit_4(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(), loss_func=MSELoss(), clip_grad=ClipValue(0.1),
                      lr_scheduler=ReduceLROnPlateau(), epochs=10)

    count = 1
    for i in trainer(*data[1]):
        assert isinstance(i, dict)
        assert i['i_epoch'] == count
        if count == 3:
            trainer.early_stop('stop')
        count += 1

    assert trainer.total_epochs == 3
    assert trainer._early_stopping == (True, 'stop')

    trainer.reset()
    train_set = DataLoader(TensorDataset(*data[1]))
    count = 1
    for i in trainer(training_dataset=train_set):
        assert isinstance(i, dict)
        assert i['i_batch'] == count
        if count == 3:
            trainer.early_stop('stop!!!')
        count += 1
    assert trainer.total_iterations == 3
    assert trainer._early_stopping == (True, 'stop!!!')


def test_persist_1(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(lr=0.1), loss_func=MSELoss(), epochs=200)
    trainer.extend(TensorConverter(), Persist('model_dir'))
    trainer.fit(*data[1], *data[1])

    persist = trainer['persist']
    checker = persist._checker
    assert isinstance(persist, Persist)
    assert isinstance(checker.model, torch.nn.Module)
    assert isinstance(checker.describe, dict)
    assert isinstance(checker.files, list)
    assert set(checker.files) == {'model', 'init_state', 'model_structure', 'describe', 'training_info', 'final_state'}

    trainer = Trainer.load(checker)
    assert isinstance(trainer.training_info, pd.DataFrame)
    assert isinstance(trainer.model, torch.nn.Module)
    assert isinstance(trainer._training_info, list)
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer.x_val is None
    assert trainer.y_val is None
    assert trainer.validate_dataset is None
    assert trainer._optimizer_state is None
    assert trainer.total_epochs == 0
    assert trainer.total_iterations == 0
    assert trainer.loss_type is None
    assert trainer.loss_func is None

    trainer = Trainer.load(from_=checker.path, optimizer=Adam(), loss_func=MSELoss(),
                           lr_scheduler=ExponentialLR(gamma=0.99),
                           clip_grad=ClipValue(clip_value=0.1))
    assert isinstance(trainer._scheduler, ExponentialLR)
    assert isinstance(trainer._optim, Adam)
    assert isinstance(trainer.clip_grad, ClipValue)
    assert isinstance(trainer.loss_func, MSELoss)


def test_trainer_prediction_1(data):
    model = deepcopy(data[0])
    trainer = Trainer(model=model, optimizer=Adam(lr=0.1), loss_func=MSELoss(), epochs=200)
    trainer.extend(TensorConverter())
    trainer.fit(*data[1], *data[1])

    trainer = Trainer(model=model).extend(TensorConverter())
    y_p = trainer.predict(data[1][0])
    assert np.any(np.not_equal(y_p, data[1][1].numpy()))
    assert np.allclose(y_p, data[1][1].numpy(), rtol=0, atol=0.2)

    y_p, y_t = trainer.predict(*data[1])
    assert np.any(np.not_equal(y_p, y_t))
    assert np.allclose(y_p, y_t, rtol=0, atol=0.2)

    val_set = DataLoader(TensorDataset(*data[1]), batch_size=50)
    y_p, y_t = trainer.predict(dataset=val_set)
    assert np.any(np.not_equal(y_p, y_t))
    assert np.allclose(y_p, y_t, rtol=0, atol=0.2)

    with pytest.raises(RuntimeError, match='parameters <x_in> and <dataset> are mutually exclusive'):
        trainer.predict(*data[1], dataset='not none')


if __name__ == "__main__":
    pytest.main()
