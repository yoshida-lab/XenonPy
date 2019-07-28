#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

__all__ = ['ClipNorm', 'ClipValue']


class ClipNorm(object):

    def __init__(self, max_norm, norm_type=2):
        r"""Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        self.norm_type = norm_type
        self.max_norm = max_norm

    def __call__(self, params):
        clip_grad_norm_(parameters=params, max_norm=self.max_norm, norm_type=self.norm_type)


class ClipValue(object):

    def __init__(self, clip_value):
        r"""Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Arguments:
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        self.clip_value = clip_value

    def __call__(self, params):
        clip_grad_value_(parameters=params, clip_value=self.clip_value)
