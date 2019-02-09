#  Copyright 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd
import pytest

from xenonpy.descriptor import Compositions


@pytest.fixture(scope='module')
def setup():
    # prepare path
    print('Test start')
    yield test
    print('Test over')


def test_comp_descripotor():
    desc = Compositions(n_jobs=1)
    ret = desc.fit_transform(pd.Series([{'H': 2}], name='composition'))


if __name__ == "__main__":
    pytest.main()
