#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

###layers
from ConvolutionLayers import ConvolutionLayer
from PoolingLayers import MaxPoolingLayer
from FeedForwardLayers import DropoutLayer
from LogisticLayers import LogisticLayer
from RegressionLayers import LinearRegressionLayer

class SimpleLogisticCNN(object):

    def __init__(self, input, target, batchsize, isTrain, dropout=False, nkerns=(30,30), \
                              colorchannels=1, imageShape=(28,28), fully_activation=T.tanh):
        self.input = input
        self.rng = np.random.RandomState(1234)
        self.img_shape=(batchsize, colorchannels, imageShape[0], imageShape[1])

        self.convLayer1 = ConvolutionLayer(
            input=self.input,
            rng=self.rng,
            image_shape=self.img_shape,
            filter_shape=(nkerns[0], colorchannels, 5, 5),
            poolsize=(2, 2)
        )
        
        self.poolingLayer1 = MaxPoolingLayer(
            input=self.convLayer1.output,
            image_shape=self.convLayer1.outshape,
            poolsize=(2, 2)
        )

        self.convLayer2 = ConvolutionLayer(
            input=self.poolingLayer1.output,
            rng = self.rng,
            image_shape=self.poolingLayer1.outshape,
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        
        self.poolingLayer2 = MaxPoolingLayer(
            input=self.convLayer2.output,
            image_shape=self.convLayer2.outshape,
            poolsize=(2, 2)
        )

        self.dropLayer1 = DropoutLayer(
            input=self.poolingLayer2.output.flatten(2),
            rng=self.rng,
            n_in=nkerns[1] * self.poolingLayer2.outshape[2] * self.poolingLayer2.outshape[3],
            n_hidden=500,
            activation=fully_activation,
            dropout=dropout,
            isTrain=isTrain
        )

        self.logisticLayer1 = LogisticLayer(
            input=self.dropLayer1.output,
            target=target,
            rng=self.rng,
            n_hidden=500, 
            n_out=10
        )

        self.params = self.convLayer1.params + self.convLayer2.params + self.dropLayer1.params + self.logisticLayer1.params
        self.cost = self.logisticLayer1.cost
        self.misclassified = self.logisticLayer1.misclassified

class SimpleLRegressionCNN(object):

    def __init__(self, input, target, batchsize, isTrain, dropout=False, nkerns=(30,30), colorchannels=1, \
                              imageShape=(28,28), fully_activation=T.tanh):
        self.input = input
        self.rng = np.random.RandomState(1234)
        self.img_shape=(batchsize, colorchannels, imageShape[0], imageShape[1])

        self.convLayer1 = ConvolutionLayer(
            input=self.input,
            rng=self.rng,
            image_shape=self.img_shape,
            filter_shape=(nkerns[0], colorchannels, 5, 5),
            poolsize=(2, 2)
        )
        
        self.poolingLayer1 = MaxPoolingLayer(
            input=self.convLayer1.output,
            image_shape=self.convLayer1.outshape,
            poolsize=(2, 2)
        )

        self.convLayer2 = ConvolutionLayer(
            input=self.poolingLayer1.output,
            rng = self.rng,
            image_shape=self.poolingLayer1.outshape,
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        
        self.poolingLayer2 = MaxPoolingLayer(
            input=self.convLayer2.output,
            image_shape=self.convLayer2.outshape,
            poolsize=(2, 2)
        )

        self.dropLayer1 = DropoutLayer(
            input=self.poolingLayer2.output.flatten(2),
            rng=self.rng,
            n_in=nkerns[1] * self.poolingLayer2.outshape[2] * self.poolingLayer2.outshape[3],
            n_hidden=500,
            activation=fully_activation,
            dropout=dropout,
            isTrain=isTrain
        )

        self.linearRegressionLayer1 = LinearRegressionLayer(
            input=self.dropLayer1.output,
            target=target,
            rng=self.rng,
            n_hidden=500
        )

        self.params = self.convLayer1.params + self.convLayer2.params + self.dropLayer1.params + self.linearRegressionLayer1.params
        self.cost = self.linearRegressionLayer1.cost