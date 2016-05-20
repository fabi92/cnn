#!/usr/bin/env python

import theano.tensor as T

def negative_log_likelihood(predictions, groundTruth):
    return -T.mean(T.log(predictions)[T.arange(groundTruth.shape[0]), groundTruth])

def mean_squared_error_loss(inp, out):
    return  T.mean((out-inp)**2)

def squared_error_loss(inp, out):
	return T.sum((out - inp)**2)

def kullback_leibler_divergence(p, q):
	avg_activations = T.mean(q, axis=1)
	return T.sum( p * T.log2( p / avg_activations ) + (1-p) * T.log2( (1-p) / ( 1 - avg_activations )))