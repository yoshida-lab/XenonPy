#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from pathlib import Path
from scipy.special import softmax

import numpy as np
import pandas as pd
import pytest
from shutil import rmtree

import torch
import os

from xenonpy.model import SequentialLinear
from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension, BaseRunner
from xenonpy.model.training.extension import TensorConverter, Validator, Persist
from xenonpy.model.utils import regression_metrics, classification_metrics


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    dir_ = os.path.dirname(os.path.abspath(__file__))

    yield

    try:
        rmtree(str(Path('.').resolve() / 'test_model'))
    except:
        pass
    try:
        rmtree(str(Path('.').resolve() / 'test_model@1'))
    except:
        pass
    try:
        rmtree(str(Path('.').resolve() / 'test_model_1'))
    except:
        pass
    try:
        rmtree(str(Path('.').resolve() / 'test_model_2'))
    except:
        pass
    try:
        rmtree(str(Path('.').resolve() / 'test_model_3'))
    except:
        pass
    try:
        rmtree(str(Path('.').resolve() / Path(os.getcwd()).name))
    except:
        pass

    print('test over')


def test_base_runner_1():
    ext = BaseExtension()
    x, y = 1, 2
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y, None) == (y, None)

    x, y = (1,), 2
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y, None) == (y, None)

    x, y = (1,), (2,)
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y, y) == (y, y)


def test_tensor_converter_1():

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.non_blocking = False

        def predict(self, x_, y_):  # noqa
            return x_, y_

    trainer = _Trainer()
    arr_1 = [1, 2, 3]
    np_1 = np.asarray(arr_1)
    se_1 = pd.Series(arr_1)
    pd_1 = pd.DataFrame(arr_1)
    np_ = np.asarray([arr_1, arr_1])
    pd_ = pd.DataFrame(np_)
    tensor_ = torch.Tensor(np_)

    # test auto reshape; #189
    converter = TensorConverter(auto_reshape=False)
    x, y = converter.input_proc(np_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3,)

    x, y = converter.input_proc(se_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3,)

    x, y = converter.input_proc(pd_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 1)

    converter = TensorConverter()
    x, y = converter.input_proc(np_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 1)

    x, y = converter.input_proc(se_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 1)

    x, y = converter.input_proc(pd_1, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 1)

    # normal tests
    x, y = converter.input_proc(np_, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(pd_, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(tensor_, None, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(np_, np_, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)

    x, y = converter.input_proc(pd_, pd_, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)

    x, y = converter.input_proc(tensor_, tensor_, trainer=trainer)  # noqa
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)

    converter = TensorConverter(x_dtype=torch.long)
    x, y = converter.input_proc((np_, np_), np_, trainer=trainer)  # noqa
    assert isinstance(x, tuple)
    assert len(x) == 2
    assert x[0].dtype == torch.long
    assert x[1].dtype == torch.long

    converter = TensorConverter(x_dtype=(torch.long, torch.float32), y_dtype=torch.long)
    x, y = converter.input_proc((np_, np_), np_, trainer=trainer)  # noqa
    assert isinstance(x, tuple)
    assert len(x) == 2
    assert x[0].dtype == torch.long
    assert x[1].dtype == torch.float32
    assert y.dtype == torch.long

    converter = TensorConverter(x_dtype=(torch.long, torch.float32))
    x, y = converter.input_proc((pd_, pd_), pd_, trainer=trainer)  # noqa
    assert isinstance(x, tuple)
    assert len(x) == 2
    assert x[0].dtype == torch.long
    assert x[1].dtype == torch.float32

    # for tensor input, dtype change will never be executed
    converter = TensorConverter(x_dtype=(torch.long, torch.long))
    x, y = converter.input_proc((tensor_, tensor_), tensor_, trainer=trainer)  # noqa
    assert isinstance(x, tuple)
    assert len(x) == 2
    assert x[0].dtype == torch.float32
    assert x[1].dtype == torch.float32


def test_tensor_converter_2():

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.non_blocking = False

        def predict(self, x_, y_):
            return x_, y_

    trainer = _Trainer()
    converter = TensorConverter()
    np_ = np.asarray([[1, 2, 3], [4, 5, 6]])
    pd_ = pd.DataFrame(np_)
    tensor_ = torch.Tensor(np_)  # noqa

    x, y = converter.input_proc(np_, np_[0], trainer=trainer)  # noqa
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
    assert torch.equal(y, tensor_[0].unsqueeze(-1))

    x, y = converter.input_proc(pd_, pd_.iloc[0], trainer=trainer)  # noqa
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
    assert torch.equal(y, tensor_[0].unsqueeze(-1))

    x, y = converter.input_proc(tensor_, tensor_[0], trainer=trainer)  # noqa
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,)
    assert torch.equal(y, tensor_[0])


def test_tensor_converter_3():
    np_ = np.asarray([[1, 2, 3], [4, 5, 6]])
    tensor_ = torch.from_numpy(np_)

    converter = TensorConverter()
    y, y_ = converter.output_proc(tensor_, None, training=True)
    assert y_ is None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 3)
    assert torch.equal(y, tensor_)

    y, y_ = converter.output_proc(tensor_, tensor_, training=True)
    assert isinstance(y, torch.Tensor)
    assert isinstance(y_, torch.Tensor)
    assert y.equal(y_)
    assert y.shape == (2, 3)
    assert torch.equal(y, tensor_)

    y, _ = converter.output_proc((tensor_,), None, training=True)
    assert isinstance(y, tuple)
    assert isinstance(y[0], torch.Tensor)
    assert torch.equal(y[0], tensor_)

    y, y_ = converter.output_proc(tensor_, tensor_, training=False)
    assert isinstance(y, np.ndarray)
    assert isinstance(y_, np.ndarray)
    assert np.all(y == y_)
    assert y.shape == (2, 3)
    assert np.all(y == tensor_.numpy())

    y, _ = converter.output_proc((tensor_,), None, training=False)
    assert isinstance(y, tuple)
    assert isinstance(y[0], np.ndarray)
    assert np.all(y[0] == tensor_.numpy())

    converter = TensorConverter(argmax=True)
    y, y_ = converter.output_proc(tensor_, tensor_, training=False)
    assert isinstance(y, np.ndarray)
    assert isinstance(y_, np.ndarray)
    assert y.shape == (2,)
    assert y_.shape == (2, 3)
    assert np.all(y == np.argmax(np_, 1))

    y, y_ = converter.output_proc((tensor_, tensor_), None, training=False)
    assert isinstance(y, tuple)
    assert y_ is None
    assert y[0].shape == (2,)
    assert y[0].shape == y[1].shape
    assert np.all(y[0] == np.argmax(np_, 1))

    converter = TensorConverter(probability=True)
    y, y_ = converter.output_proc(tensor_, tensor_, training=False)
    assert isinstance(y, np.ndarray)
    assert isinstance(y_, np.ndarray)
    assert y.shape == (2, 3)
    assert y_.shape == (2, 3)
    assert np.all(y == softmax(np_, 1))

    y, y_ = converter.output_proc((tensor_, tensor_), None, training=False)
    assert isinstance(y, tuple)
    assert y_ is None
    assert y[0].shape == (2, 3)
    assert y[0].shape == y[1].shape
    assert np.all(y[0] == softmax(np_, 1))


def test_validator_1():
    x = np.random.randn(100)  # input
    y = x + np.random.rand() * 0.001  # true values

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.x_val = x
            self.y_val = y
            self.loss_type = 'train_loss'

        def predict(self, x_, y_):
            return x_, y_

    val = Validator('regress', each_iteration=False)

    step_info = OrderedDict(train_loss=0, i_epoch=0)
    val.step_forward(trainer=_Trainer(), step_info=step_info)  # noqa
    assert 'val_mae' not in step_info

    step_info = OrderedDict(train_loss=0, i_epoch=1)
    val.step_forward(trainer=_Trainer(), step_info=step_info)  # noqa
    assert step_info['val_mae'] == regression_metrics(y, x)['mae']
    assert set(step_info.keys()) == {
        'i_epoch', 'val_mae', 'val_mse', 'val_rmse', 'val_r2', 'val_pearsonr', 'val_spearmanr',
        'val_p_value', 'val_max_ae', 'train_loss'
    }


def test_validator_2():
    y = np.random.randint(3, size=10)  # true labels
    x = np.zeros((10, 3))  # input
    for i, j in enumerate(y):
        x[i, j] = 1

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.x_val = x
            self.y_val = y
            self.loss_type = 'train_loss'

        def predict(self, x_, y_):  # noqa
            return x_, y_

    val = Validator('classify', each_iteration=False)

    step_info = OrderedDict(train_loss=0, i_epoch=0)
    val.step_forward(trainer=_Trainer(), step_info=step_info)  # noqa
    assert 'val_f1' not in step_info

    step_info = OrderedDict(train_loss=0, i_epoch=1)
    val.step_forward(trainer=_Trainer(), step_info=step_info)  # noqa
    assert step_info['val_f1'] == classification_metrics(y, x)['f1']
    assert set(step_info.keys()) == {
        'i_epoch', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall', 'val_macro_f1',
        'val_macro_precision', 'val_macro_recall', 'train_loss'
    }


def test_persist_1(data):

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.model = SequentialLinear(50, 2)

        def predict(self, x_, y_):  # noqa
            return x_, y_

    p = Persist()

    with pytest.raises(ValueError, match='can not access property `path` before training'):
        p.path

    p.before_proc(trainer=_Trainer())
    assert p.path == str(Path('.').resolve() / Path(os.getcwd()).name)
    with pytest.raises(ValueError, match='can not reset property `path` after training'):
        p.path = 'aa'

    p = Persist('test_model')
    p.before_proc(trainer=_Trainer())
    assert p.path == str(Path('.').resolve() / 'test_model')
    assert (Path('.').resolve() / 'test_model' / 'describe.pkl.z').exists()
    assert (Path('.').resolve() / 'test_model' / 'init_state.pth.s').exists()
    assert (Path('.').resolve() / 'test_model' / 'model.pth.m').exists()
    assert (Path('.').resolve() / 'test_model' / 'model_structure.pkl.z').exists()

    p = Persist('test_model', increment=True)
    p.before_proc(trainer=_Trainer())
    assert p.path == str(Path('.').resolve() / 'test_model@1')
    assert (Path('.').resolve() / 'test_model@1' / 'describe.pkl.z').exists()
    assert (Path('.').resolve() / 'test_model@1' / 'init_state.pth.s').exists()
    assert (Path('.').resolve() / 'test_model@1' / 'model.pth.m').exists()
    assert (Path('.').resolve() / 'test_model@1' / 'model_structure.pkl.z').exists()


def test_persist_save_checkpoints(data):

    class _Trainer(BaseRunner):

        def __init__(self):
            super().__init__()
            self.model = SequentialLinear(50, 2)

        def predict(self, x_, y_):  # noqa
            return x_, y_

    cp_1 = Trainer.checkpoint_tuple(
        id='cp_1',
        iterations=111,
        model_state=SequentialLinear(50, 2).state_dict(),
    )
    cp_2 = Trainer.checkpoint_tuple(
        id='cp_2',
        iterations=111,
        model_state=SequentialLinear(50, 2).state_dict(),
    )

    # save checkpoint
    p = Persist('test_model_1', increment=False, only_best_states=False)
    p.before_proc(trainer=_Trainer())
    p.on_checkpoint(cp_1, trainer=_Trainer())
    p.on_checkpoint(cp_2, trainer=_Trainer())
    assert (Path('.').resolve() / 'test_model_1' / 'checkpoints' / 'cp_1.pth.s').exists()
    assert (Path('.').resolve() / 'test_model_1' / 'checkpoints' / 'cp_2.pth.s').exists()

    # reduced save checkpoint
    p = Persist('test_model_2', increment=False, only_best_states=True)
    p.before_proc(trainer=_Trainer())
    p.on_checkpoint(cp_1, trainer=_Trainer())
    p.on_checkpoint(cp_2, trainer=_Trainer())
    assert (Path('.').resolve() / 'test_model_2' / 'checkpoints' / 'cp.pth.s').exists()
    assert not (Path('.').resolve() / 'test_model_2' / 'checkpoints' / 'cp_1.pth.s').exists()
    assert not (Path('.').resolve() / 'test_model_2' / 'checkpoints' / 'cp_2.pth.s').exists()

    # no checkpoint will be saved
    p = Persist('test_model_3', increment=False, only_best_states=True)
    p.before_proc(trainer=_Trainer())
    p.on_checkpoint(cp_2, trainer=_Trainer())
    assert not (Path('.').resolve() / 'test_model_3' / 'checkpoints' / 'cp.pth.s').exists()
    assert not (Path('.').resolve() / 'test_model_3' / 'checkpoints' / 'cp_2.pth.s').exists()


if __name__ == "__main__":
    pytest.main()
