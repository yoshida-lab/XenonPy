#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

__all__ = ['NLLLoss', 'NLLLoss2d', 'L1Loss', 'MSELoss', 'CrossEntropyLoss', 'CTCLoss', 'PoissonNLLLoss', 'KLDivLoss',
           'BCELoss', 'BCEWithLogitsLoss', 'MarginRankingLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss',
           'SmoothL1Loss', 'SoftMarginLoss', 'MultiLabelSoftMarginLoss', 'CosineEmbeddingLoss', 'MultiMarginLoss',
           'TripletMarginLoss']

from torch.nn.modules.loss import *
