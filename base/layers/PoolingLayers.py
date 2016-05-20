#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

class MaxPoolingLayer(object):
    
    def __init__(self, input, poolsize=(2, 2)):
        self.input = input

        self.output = downsample.max_pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )