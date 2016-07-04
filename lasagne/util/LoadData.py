#!/usr/bin/env python

import theano
import cPickle as pickle
import os
import numpy as np
import h5py

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

def shared_dataset(data, labels, labelstype='int32', borrow=True):
    shared_x = theano.shared(np.asarray(data,
                                        dtype=theano.config.floatX),
                                        name='imageData',
                                        borrow=borrow)
    shared_y = theano.shared(np.asarray(labels,
                                        dtype=theano.config.floatX),
                                        name='imageLabels',
                                        borrow=borrow)
    return shared_x, theano.tensor.cast(shared_y, labelstype)

def performAugmentation(train_set_x, train_set_y, shuffle=True):
    try:
        import scipy.ndimage.interpolation as interp
    except ImportError:
        print("Could Not Import 'scipy.ndimage.interpolation'\nMoving On With Unaugmented Data")
        return train_set_x, train_set_y
    x = np.zeros((train_set_x.shape[0]*2,1,105,111))
    y = np.zeros((train_set_y.shape[0]*2,train_set_y.shape[1]))
    x[:train_set_x.shape[0]] = train_set_x[:]
    y[:train_set_y.shape[0]] = train_set_y[:]

    for nIdx in range(train_set_x.shape[0]):
        new = np.zeros((105,111))
        new[:,:] = train_set_x[nIdx,0,:,:]
        if np.random.randint(2) == 1: ##left right flip
            new = np.fliplr(new)
        if np.random.randint(2) == 1: ##left right flip
            new = np.flipud(new)
        ##random rotate
        new = interp.rotate(new,np.random.rand()*360, reshape=False)
        x[nIdx+train_set_x.shape[0],0,:,:] = new[:,:]
        y[nIdx+train_set_y.shape[0],:] = train_set_y[nIdx,:]

    if shuffle:
        img_lables = zip(x, y)
        np.random.shuffle(img_lables)
        x, y = zip(*img_lables)
        x = np.array(x)
        y = np.array(y)
    return x, y

def loadFingernails(path, shared=True, context="forces", augment=False, shuffleAugment=True):
    fingernails = h5py.File(path,'r')

    train_set_x = np.array(fingernails.get('frames_trn')).T / 255.
    train_set_x = train_set_x.reshape(train_set_x.shape[0],1,105,111)
    train_set_y = np.array(fingernails.get('forces_trn')).T
    ### valid set ihas more images than labels
    valid_set_x = np.array(fingernails.get('frames_val')).T / 255.
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0],1,105,111)[:5000]
    valid_set_y = np.array(fingernails.get('forces_val')).T[:5000]
    test_set_x = np.array(fingernails.get('frames_tst')).T / 255.
    test_set_x = test_set_x.reshape(test_set_x.shape[0],1,105,111)
    test_set_y = np.array(fingernails.get('forces_tst')).T

    train_set_x = train_set_x.astype('float32')
    valid_set_x = valid_set_x.astype('float32')
    test_set_x = test_set_x.astype('float32')


    train_set_y = train_set_y.astype('float32')
    valid_set_y = valid_set_y.astype('float32')
    test_set_y = test_set_y.astype('float32')

    # #####for fast testing we only take a look at force_x
    # train_set_x = train_set_x[:500]
    if context == "forces":
        train_set_y = train_set_y[:,:3]
        valid_set_y = valid_set_y[:,:3]
        test_set_y = test_set_y[:,:3]
    elif context == "torques":
        train_set_y = train_set_y[:,3:6]
        valid_set_y = valid_set_y[:,3:6]
        test_set_y = test_set_y[:,3:6]
    elif context == "surface":
        train_set_y = train_set_y[:,6:8]
        valid_set_y = valid_set_y[:,6:8]
        test_set_y = test_set_y[:,6:8]

    if shared:
        test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y, labelstype='float32')
        valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y, labelstype='float32')
        train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y, labelstype='float32')

    if augment:
        print('Augmenting The Training Data')
        train_set_x, train_set_y = performAugmentation(train_set_x, train_set_y, shuffleAugment)

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)