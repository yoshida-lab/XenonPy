#  Copyright (c) 2021. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import pytest


def test_import_loss():
    with pytest.raises(ImportError):
        import xenonpy.model.training.loss

def test_import_lr_scheduler():
    with pytest.raises(ImportError):
        import xenonpy.model.training.lr_scheduler

def test_import_optimizer():
    with pytest.raises(ImportError):
        import xenonpy.model.training.optimizer


if __name__ == "__main__":
    pytest.main()
