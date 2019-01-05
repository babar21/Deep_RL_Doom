#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils for nn buidling, optimization
"""

from keras import backend as K
import tensorflow as tf
import keras.optimizers as kopt

#%% DFP specific functions

def normalize_layer(x):
    """
    DFP : subtract the mean along the action axis for the DFP network
    """
    m = K.mean(x, axis=0, keepdims=True)
    return x-m
    

#%% Optimization

def get_optimizer(optimizer_params):
    """
    Parse optimizer parameters.
    Input : dict with key 
            'type' -> optimizer type
            other -> optimizer legit parameters
    """
    if 'type' in optimizer_params:
        method = optimizer_params.pop('type', None)
    else :
        raise Exception('You must provide an optimization method')

    if method == 'adadelta':
        optim_fn = kopt.Adadelta(**optimizer_params)
    elif method == 'adagrad':
        optim_fn = kopt.Adagrad(**optimizer_params)
    elif method == 'adam':
        optim_fn = kopt.Adam(**optimizer_params)
    elif method == 'adamax':
        optim_fn = kopt.Adamax(**optimizer_params)
    elif method == 'sgd':
        optim_fn = kopt.SGD(**optimizer_params)
    elif method == 'rmsprop':
        optim_fn = kopt.RMSprop(**optimizer_params)
    else :
        raise Exception('Unknown optimization method: "%s"' % method)

    return optim_fn