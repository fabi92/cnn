#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np

from Cost import mean_squared_error_loss, squared_error_loss

class LinearRegressionLayer(object):
    def __init__(self, input, target, rng, n_hidden):
        self.input = input
        self.target = target

        self.weights = theano.shared(
            value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_hidden)),
                        high=np.sqrt(6. / (n_hidden)),
                        size=(n_hidden)
                    )
            ),
            name='weights',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=0.0,
            name='bias',
            borrow=True
        )

        self.regression = T.dot(self.input, self.weights) + self.bias
        self.cost = squared_error_loss(self.regression, self.target)

        self.params = [self.weights, self.bias]