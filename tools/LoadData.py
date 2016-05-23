#!/usr/bin/env python

import theano
import cPickle as pickle
import os
import numpy as np

def loadMNIST(dataset, shared=True):
    with open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    if not shared:
        return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)

    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)

def loadMNISTConcatenated(dataset, shared=True):
    (train_images, train_labels), (validation_images, validation_labels), \
         (test_images, test_labels) = loadMNIST(dataset, False)

    fullSetImgs = np.concatenate((train_images, validation_images, test_images), axis=0)
    fullSetLabels = np.concatenate((train_labels, validation_labels, test_labels), axis=0)
    
    if shared:
        return shared_dataset(fullSetImgs, fullSetLabels)
    
    return (fullSetImgs, fullSetLabels)

def loadCIFAR10Color(path, shared=True):
    images, labels = loadCIFAR10ColorConcatenated(path, shared=False)
    if shared:
        train_set_x, train_set_y = shared_dataset(images[:40000], labels[:40000])
        valid_set_x, valid_set_y = shared_dataset(images[40000:50000], labels[40000:50000])
        test_set_x, test_set_y = shared_dataset(images[50000:], labels[50000:])
        return (train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)
    else: 
        return ((images[:40000], labels[:40000]), (images[40000:50000], labels[40000:50000]), (images[50000:], labels[50000:]))


def loadCIFAR10ColorConcatenated(path, shared=True):
    labels = np.zeros(60000, dtype='int32')
    images = np.zeros((60000, 3, 32, 32), dtype='float64')
    
    for subdir, dirs, files in os.walk(path):
        n_loaded = 0
        for file in files:
            if file.startswith("data"): 
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()

                assert data['data'].dtype == np.uint8
                images[n_loaded:n_loaded + 10000] = data['data'].reshape(10000, 3,32,32)
                labels[n_loaded:n_loaded + 10000] = data['labels']
                n_loaded += 10000
            elif file.startswith('test'):
                filepath = subdir + os.sep + file
                fo = open(filepath, 'rb')
                data = pickle.load(fo)
                fo.close()
                assert data['data'].dtype == np.uint8
                images[50000:60000] = data['data'].reshape(10000, 3,32,32)
                labels[50000:60000] = data['labels']
    if shared:
        images, labels = shared_dataset(images, labels)
    return images, labels

def shared_dataset(data, labels, borrow=True):
    shared_x = theano.shared(np.asarray(data,
                                        dtype=theano.config.floatX),
                                        name='imageData',
                                        borrow=borrow)
    shared_y = theano.shared(np.asarray(labels,
                                        dtype=theano.config.floatX),
                                        name='imageLabels',
                                        borrow=borrow)
    return shared_x, theano.tensor.cast(shared_y, 'int32')