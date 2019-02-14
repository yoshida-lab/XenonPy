#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd
import pytest

from xenonpy.descriptor import Compositions


def test_comp_descripotor():
    desc = Compositions(n_jobs=1)
    desc.fit_transform(pd.Series([{'H': 2}], name='composition'))


if __name__ == "__main__":
    pytest.main()
