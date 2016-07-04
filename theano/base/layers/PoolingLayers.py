#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

class MaxPoolingLayer(object):
    
    def __init__(self, input, image_shape, poolsize=(2, 2)):
        self.input = input

        self.output = downsample.max_pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )
        self.outshape=(image_shape[0], image_shape[1], image_shape[2] / poolsize[0], image_shape[3] / poolsize[1])