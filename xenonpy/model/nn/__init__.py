# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import warnings as warnings

with warnings.catch_warnings():
    # warnings.simplefilter('default')
    try:
        import torch
    except ImportError:
        warnings.warn("Can't find pytorch, will not load neural network modules.", RuntimeWarning)
    else:
        from .checker import Checker
        from .runner import ModelRunner
        from .model_maker import Generator1d, Sequential
        from .layer import Layer1d
        from . import wrap
