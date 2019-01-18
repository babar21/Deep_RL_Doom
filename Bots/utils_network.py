#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils for nn buidling, optimization
"""

from keras import backend as K
import keras.optimizers as kopt
import h5py
import pickle

#%% DFP specific functions

def normalize_layer(x):
    """
    DFP : subtract the mean along the action axis for the DFP network
    """
    m = K.mean(x, axis=1, keepdims=True)
    return x-m
    

#%% Optimization

def get_optimizer(optimizer_params):
    """
    Parse optimizer parameters.
    Input : dict with key 
            'type' -> optimizer type
            other -> optimizer legit parameters
    """
    optimizer_param = optimizer_params.copy()
    if 'type' in optimizer_param:
        method = optimizer_param.pop('type', None)
    else :
        raise Exception('You must provide an optimization method')

    if method == 'adadelta':
        optim_fn = kopt.Adadelta(**optimizer_param)
    elif method == 'adagrad':
        optim_fn = kopt.Adagrad(**optimizer_param)
    elif method == 'adam':
        optim_fn = kopt.Adam(**optimizer_param)
    elif method == 'adamax':
        optim_fn = kopt.Adamax(**optimizer_param)
    elif method == 'sgd':
        optim_fn = kopt.SGD(**optimizer_param)
    elif method == 'rmsprop':
        optim_fn = kopt.RMSprop(**optimizer_param)
    else :
        raise Exception('Unknown optimization method: "%s"' % method)

    return optim_fn

#%% saving
    
def saving_stats(nb_episode, stats, network, name):
    """
    saving weights and stats from the experiment
    """
    # game stats
    with open('{}_game_stats_{}'.format(name,nb_episode),'wb') as fp:
        pickle.dump(stats,fp)
    
    # network
    network.save('{}_network_{}'.format(name,nb_episode))
