# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import pytest

from xenonpy.model.nn import BaseRunner


def test_base_runner():
    with BaseRunner() as runner:
        assert hasattr(runner, '__enter__')
        assert hasattr(runner, '__exit__')


if __name__ == "__main__":
    pytest.main()
