## imports
from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle
import argparse

import numpy as np
import theano
import theano.tensor as T
import lasagne

from sklearn.metrics import r2_score

from Networks import buildResNet
from util.Exporter import csvExportHeader

class ResNet(object):

    def __init__(self, input_var, target_var, inshape=(105,111), n_predictions=3, blocks_length=3, filter_count=16):
        self.cnn = buildResNet(input_var, inshape, n_predictions, blocks_length, filter_count)
        self.cropShape = inshape

        self.input_var = input_var
        self.target_var = target_var

        self.trainRes = [[],[],[],[]]
        self.valRes = [[],[],[],[]]
        self.testRes = [[],[],[],[]]

    def compile(self, lr_rate):
        prediction = lasagne.layers.get_output(self.cnn)
        tr_mse = lasagne.objectives.squared_error(prediction, self.target_var).mean()
        tr_mae = (abs(prediction - self.target_var)).mean()
        all_layers = lasagne.layers.get_all_layers(self.cnn)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = tr_mse + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(self.cnn, trainable=True)

        sh_lr = theano.shared(lasagne.utils.floatX(lr_rate))
        updates = lasagne.updates.momentum(
                loss, 
                params, 
                learning_rate=sh_lr, 
                momentum=0.9
        )

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function(
            [self.input_var, self.target_var],
            [tr_mse, tr_mae, prediction, self.target_var],
            updates=updates
        )

        # Create a loss expression for validation/testing
        test_prediction = lasagne.layers.get_output(self.cnn, deterministic=True)
        tst_mse = lasagne.objectives.squared_error(test_prediction, self.target_var).mean()
        tst_mae = (abs(test_prediction - self.target_var)).mean()

        # Compile a second function computing the validation loss and accuracy:
        self.val_fn = theano.function(
            [self.input_var, self.target_var],
            [tst_mse, tst_mae, test_prediction, self.target_var]
        )

    def train(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,  epochs, exportLive=False, exportTxtFile="test.csv"):
        self.trainRes = [[],[],[],[]]
        self.valRes = [[],[],[],[]]
        self.testRes = [[],[],[],[]]

        idx = 0
        # launch the training loop
        print("Starting training...")

        if exportLive:
            csvExportHeader(exportTxtFile)
        # We iterate over epochs:

        for epoch in range(epochs):
            # shuffle training data
            train_indices = np.arange(len(X_train))
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_mse = 0
            train_mae = 0
            train_pred = []
            train_labels = []
            train_batches = 0
            start_time = time.time()

            for batch in self.iterate_minibatches(X_train, Y_train, 20, shuffle=True, augment=True, cropShape=self.cropShape):
                inputs, targets = batch
                mse,mae,pred,lab = self.train_fn(inputs, targets)
                train_mse += mse
                train_mae += mae
                train_pred = train_pred + pred.tolist()
                train_labels = train_labels + lab.tolist()
                train_batches += 1
                idx +=1

            # And a full pass over the validation data:
            val_mse = 0
            val_mae = 0
            val_pred = []
            val_labels = []
            val_batches = 0

            for batch in self.iterate_minibatches(X_valid, Y_valid, 20, shuffle=False, cropShape=self.cropShape):
                inputs, targets = batch
                mse,mae,pred,lab = self.val_fn(inputs, targets)
                val_mse += mse
                val_mae += mae
                val_pred = val_pred + pred.tolist()
                val_labels = val_labels + lab.tolist()
                val_batches += 1

            test_mse = 0
            test_mae = 0
            test_pred = []
            test_labels = []
            test_batches = 0

            for batch in self.iterate_minibatches(X_test, Y_test, 20, shuffle=False, cropShape=self.cropShape):
                inputs, targets = batch
                mse,mae,pred,lab = self.val_fn(inputs, targets)
                test_mse += mse
                test_mae += mae
                test_pred = test_pred + pred.tolist()
                test_labels = test_labels + lab.tolist()
                test_batches += 1

            self.trainRes[0].append(epoch)
            self.valRes[0].append(epoch)
            self.testRes[0].append(epoch)

            self.trainRes[1].append(train_mse / train_batches)
            self.valRes[1].append(val_mse / val_batches)
            self.testRes[1].append(test_mse / test_batches)

            self.trainRes[2].append(train_mae / train_batches)
            self.valRes[2].append(val_mae / val_batches)
            self.testRes[2].append(test_mae / test_batches)

            self.trainRes[3].append(r2_score(train_labels, train_pred))
            self.valRes[3].append(r2_score(val_labels, val_pred))
            self.testRes[3].append(r2_score(test_labels, test_pred))

            if exportLive:
                trainResTxT = [(train_mse / train_batches), (train_mae / train_batches), (r2_score(train_labels, train_pred))]
                valResTxT = [(val_mse / val_batches), (val_mae / val_batches), (r2_score(val_labels, val_pred))]
                testResTxT = [(test_mse / test_batches), (test_mae / test_batches), (r2_score(test_labels, test_pred))]
                csvExportEpoch(epoch+1, trainResTxT, valResTxT, testResTxT, exportTxtFile)

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_mse / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_mse / val_batches))
            print("  test loss:\t\t{:.6f}".format(test_mse / test_batches))


            ##learning warm up over
            if epoch == 2:
                new_lr = sh_lr.get_value() * 10.0
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (idx+1) == int(32e3) or (idx+1) == int(48e3):
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))


    ## Iterate Function for the Resnet
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False, augment=False, downsample=1, cropShape=(105,111)):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if augment:
                # as in paper : 
                # pad feature arrays with 4 pixels on each side
                # and do random cropping of 32x32
                padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
                random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
                crops = np.random.random_integers(0,high=8,size=(batchsize,2))
                for r in range(batchsize):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+cropShape[0]),crops[r,1]:(crops[r,1]+cropShape[1])]
                inp_exc = random_cropped
            else:
                inp_exc = inputs[excerpt]

            yield inp_exc, targets[excerpt]