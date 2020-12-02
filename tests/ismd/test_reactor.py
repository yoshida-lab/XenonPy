#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
from xenonpy.contrib.ismd.reactor import smi_tokenizer, SMILESInvalidError


def test_SMILESInvalidError():
    with pytest.raises(SMILESInvalidError):
        smi_tokenizer("ABCDE")
