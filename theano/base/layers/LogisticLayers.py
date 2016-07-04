#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np

from Cost import negative_log_likelihood

class LogisticLayer(object):
    def __init__(self, input, target, rng, n_hidden, n_out):
        self.input = input
        self.target = target

        self.weights = theano.shared(
            value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_hidden + n_out)),
                        high=np.sqrt(6. / (n_hidden + n_out)),
                        size=(n_hidden, n_out)
                    )
            ),
            name='weights',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bias',
            borrow=True
        )

        self.predictedHotOne = T.nnet.softmax(T.dot(self.input, self.weights) + self.bias)
        self.predictions = T.argmax(self.predictedHotOne, axis=1)
        self.cost = negative_log_likelihood(self.predictedHotOne, self.target)
        self.misclassified = T.mean(T.neq(self.predictions, self.target))

        self.params = [self.weights, self.bias]