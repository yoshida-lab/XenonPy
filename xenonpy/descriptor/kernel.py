# Copyright 2021 TsumiNa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

__all__ = ['RBFKernel']


class RBFKernel():

    def __init__(self, sigma):
        self._sigma = sigma

    def __call__(self, x_i: np.ndarray, x_j: np.ndarray):
        # K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        return np.exp(-(x_i[:, :, np.newaxis] - x_j).reshape(x_i.shape[0], -1)**2 / (2 * self._sigma**2))
