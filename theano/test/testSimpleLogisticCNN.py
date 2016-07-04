#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import argparse

from LoadData import loadMNIST, loadCIFAR10Color
from Networks import SimpleLogisticCNN

activation_functions = {
        "tanh": T.tanh,
        "relu": T.nnet.relu,
        "sigmoid": T.nnet.sigmoid,
   }

def train_network():
    ## parses the provided parameters according to the command line input
    parser = argparse.ArgumentParser(prog='Convolutional Neural Network', conflict_handler='resolve')
    
    parser.add_argument('-l', '--learningrate', type=float, default=1e-3, required=False, help='The Learning Rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.95, required=False, help='The Momentum Rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, required=False, help='The Weight Decay Rate')
    parser.add_argument('-b', '--batchsize', type=int, default=20, required=False, help='Batch Size For Training')
    parser.add_argument('-o', '--output', type=str, default="out", required=False, help='Path To The Output Folder')
    parser.add_argument('-e', '--epochs', type=int, default=500, required=False, help='Number Of Epochs')
    parser.add_argument('-a', '--activation', type=str, default="tanh", choices=["tanh", "sigmoid", "relu"]
        , required=False, help='Activation Functions For The Perceptrons [tanh/sigmoid/relu]')  
    parser.add_argument('-d', '--dropout', type=bool, default=True, required=False, help='Set To True In Order To Activate Dropout')


    requiredNamed = parser.add_argument_group('required Arguments')
    requiredNamed.add_argument('-p', '--path', type=str, required=True, help='Path To The Training Set')
    requiredNamed.add_argument('-d', '--dataset', type=str, choices=["mnist", "cifar"], required=True, help='Path To The Training Set')
   
    parsed = parser.parse_args()

    if parsed.dataset == "mnist":
        (train_images, train_labels), (validation_images, validation_labels), \
                (test_images, test_labels) = loadMNIST(parsed.path)
        imageShape=(28, 28)
        n_colors=1
    elif parsed.dataset == "cifar":
        (train_images, train_labels), (validation_images, validation_labels), \
                (test_images, test_labels) = loadCIFAR10Color(parsed.path)
        imageShape=(32, 32)
        n_colors=3
    else:
        raise ValueError('Dataset Input Is Not Valud')

    number_train_images_batches = train_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_validation_images_batches = validation_images.get_value(borrow=True).shape[0] // parsed.batchsize
    number_test_images_batches = test_images.get_value(borrow=True).shape[0] // parsed.batchsize

    index = T.lscalar() 
    imageData = T.tensor4('imageData')
    imageLabels = T.ivector('imageLabels')
    isTrain = T.iscalar('is_train')

    cnn = SimpleLogisticCNN(
        input=imageData,
        target=imageLabels,
        batchsize=parsed.batchsize,
        fully_activation=activation_functions[parsed.activation],
        nkerns=(30,30),
        colorchannels=n_colors,
        imageShape=imageShape,
        isTrain=isTrain,
        dropout=parsed.dropout
    )

    #### Momentum + Weight Decay #####
    assert parsed.momentum >= 0. and parsed.momentum < 1.
    updates = []
    for param in  cnn.params:
        last_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        delta = -parsed.learningrate * T.grad(cnn.cost, param) + parsed.momentum * last_update - \
                    parsed.learningrate * parsed.weight_decay * param
        updates.append((param, param + delta))
        updates.append((last_update, delta))

    train = theano.function(
        inputs=[index],
        outputs= cnn.cost,
        updates= updates,
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](1)
        }
    )

    validate = theano.function(
        inputs=[index],
        outputs=cnn.misclassified,
        givens={
            imageData: train_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: train_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](0)
        }
    )

    test = theano.function(
        inputs=[index],
        outputs=cnn.misclassified,
        givens={
            imageData: validation_images[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            imageLabels: validation_labels[index * parsed.batchsize: (index + 1) * parsed.batchsize],
            isTrain: np.cast['int32'](0)
        }
    )

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995 
    best_validation_loss = np.inf
    best_validation_epoch = 0
    best_testing_loss = np.inf
    best_testing_epoch = 0


    done_looping = False
    val_freq = min(number_train_images_batches, patience // 2)
    epoch = 0

    while (epoch < parsed.epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(number_train_images_batches):
            train(minibatch_index)
            idx = (epoch - 1) * number_train_images_batches + minibatch_index

            if (idx + 1) % val_freq == 0:
                val_loss = np.mean([validate(currentValidationBatch)
                                     for currentValidationBatch in range(number_validation_images_batches)])
                print("Epoch %d, Batch Index: %d / %d, Accuracy wrt Validation Samples: %f" \
                    % (epoch, minibatch_index,  number_train_images_batches, (100 - val_loss * 100)))       
                
                if val_loss < best_validation_loss:
                    if val_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, idx * patience_increase)
                        best_validation_epoch = epoch

                    best_validation_loss = val_loss
                    test_loss = np.mean([test(currentTestBatch)
                                   for currentTestBatch in range(number_test_images_batches)])
                    print("\tEpoch %d, Batch Index: %d / %d, Accuracy wrt Test Samples: %f" \
                        % (epoch, minibatch_index,  number_train_images_batches, (100 - test_loss * 100)))
                    if test_loss < best_testing_loss:
                        print('\t\tNew Best Test Result')
                        best_testing_loss = test_loss
                        best_testing_epoch = epoch

            if patience <= idx:
                done_looping = True
                break
   

    print('Optimization complete with best test score of %f %%,'
             % (100 - best_testing_loss * 100.))

    
if __name__ == '__main__':
    train_network()