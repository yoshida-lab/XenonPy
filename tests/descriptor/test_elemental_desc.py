# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest

from xenonpy.descriptor import CompositionDescriptor
from xenonpy.model.nn import Layer1d


@pytest.fixture(scope='module')
def setup():
    # prepare path
    name = 'test_checker'
    default = Path.home() / '.xenonpy/usermodel'
    dot = Path()
    test = dict(
        name=name,
        cp=dict(model_state=1, b=2),
        path='./',
        default=str(default),
        dot=str(dot),
        model=Layer1d(10, 1),
        np=np.ones((2, 3)),
        df=pd.DataFrame(np.ones((2, 3)))
    )

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield test

    dirs_ = [d for d in default.iterdir() if d.match(name + '@*')] + \
            [d for d in dot.iterdir() if d.match(name + '@*')]
    for d in dirs_:
        rmtree(str(d))
    print('test over')


def test_comp_descripotor(setup):
    desc = CompositionDescriptor(n_jobs=1)
    ret = desc.fit_transform(pd.Series([{'H': 2}], name='composition'))


if __name__ == "__main__":
    pytest.main()
