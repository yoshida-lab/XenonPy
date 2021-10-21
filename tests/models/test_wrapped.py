#  Copyright (c) 2021. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import pytest


def test_import_loss():
    try:
        import xenonpy.model.training.loss
    except ImportError:
        assert "should not raise ImportError"
        

def test_import_lr_scheduler():
    try:
        import xenonpy.model.training.lr_scheduler
    except ImportError:
        assert "should not raise ImportError"

def test_import_optimizer():
    try:
        import xenonpy.model.training.optimizer
    except ImportError:
        assert "should not raise ImportError"


if __name__ == "__main__":
    pytest.main()
