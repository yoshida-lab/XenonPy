#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from xenonpy.contrib.foo.descriptor import hello_contrib


def test_foo_1():
    assert hello_contrib() == 'Hello contribution!'
