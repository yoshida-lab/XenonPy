#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict

import pytest

from xenonpy.model.training.base import BaseRunner
from xenonpy.model.training.extension.base import BaseExtension


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    class Ext1(BaseExtension):
        def __init__(self):
            super().__init__()
            self.before = None
            self.after = None

        def before_proc(self, train: bool = True) -> None:
            self.before = 'ext1'

        def after_proc(self, train: bool = True) -> None:
            self.after = 'ext1'

        def step_forward(self, step_info) -> None:
            step_info['ext1'] = 'ext1'

        def input_proc(self, x_in, y_in=None, train: bool = True):
            if y_in is None:
                return x_in * 10, y_in
            return x_in * 10, y_in * 10

        def output_proc(self, y_pred, train: bool = True):
            return y_pred * 10

    class Ext2(BaseExtension):
        def __init__(self):
            super().__init__()
            self.before = None
            self.after = None

        def before_proc(self, train: bool = True, *, ext1=None) -> None:
            self.before = ext1.before + '_ext2'

        def after_proc(self, train: bool = True, *, ext1=None) -> None:
            self.after = ext1.after + '_ext2'

        def step_forward(self, step_info, *, ext1=None) -> None:
            step_info['ext2'] = step_info['ext1'] + '_ext2'

        def input_proc(self, x_in, y_in=None, train: bool = True):
            if y_in is None:
                return x_in * 2, y_in
            return x_in * 2, y_in * 2

        def output_proc(self, y_pred, train: bool = True):
            return y_pred * 2

    yield Ext1, Ext2
    print('test over')


def test_base_runner_1(data):
    assert BaseRunner.check_cuda(False).type == 'cpu'
    assert BaseRunner.check_cuda('cpu').type == 'cpu'

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda(True)

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda('cuda')

    with pytest.raises(RuntimeError, match='wrong device identifier'):
        BaseRunner.check_cuda('other illegal')


def test_base_runner_2(data):
    x, y = 1, 2
    runner = BaseRunner()
    assert runner.input_proc(x, y) == (x, y)


def test_base_runner_3(data):
    x, y = 1, 2
    ext1, ext2 = data[0](), data[1]()
    runner = BaseRunner()
    runner.extend(ext1, ext2)

    runner._before_proc()
    assert ext1.before == 'ext1'
    assert ext2.before == 'ext1_ext2'

    runner._after_proc()
    assert ext1.after == 'ext1'
    assert ext2.after == 'ext1_ext2'

    step_info = OrderedDict()
    runner._step_forward(step_info)
    assert step_info['ext1'] == 'ext1'
    assert step_info['ext2'] == 'ext1_ext2'

    assert len(runner._extensions) == 2
    assert runner.input_proc(x, y) == (x * 10 * 2, y * 10 * 2)
    assert runner.input_proc(x) == x * 10 * 2
    assert runner.output_proc(y) == y * 10 * 2


if __name__ == "__main__":
    pytest.main()
