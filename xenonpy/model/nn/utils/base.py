#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from collections import OrderedDict
from typing import Union

import torch


class BaseExtension(object):
    def run(self, step_info: OrderedDict, trainer):
        raise NotImplementedError()


def check_cuda(cuda: Union[bool, str, torch.device]) -> torch.device:
    if isinstance(cuda, bool):
        if cuda:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                raise RuntimeError('could not use CUDA on this machine')
        else:
            return torch.device('cpu')

    if isinstance(cuda, str):
        if 'cuda' in cuda:
            if torch.cuda.is_available():
                return torch.device(cuda)
            else:
                raise RuntimeError('could not use CUDA on this machine')
        elif 'cpu' in cuda:
            return torch.device('cpu')
        else:
            raise RuntimeError('wrong device identifier'
                               'see also: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device')

    if isinstance(cuda, torch.device):
        return cuda
