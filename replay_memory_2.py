#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Memory class
Everything needed for to store and sample from the replay memory
    
"""

import numpy as np


class ReplayMemory:

    def __init__(self, max_size, screen_shape, type_network, n_variables=0, n_features=0, n_goals=0):
        """
        Create the replay memory class
        Inputs : 
            max_size : maximum number of samples
            screen_shape : 
            n_variables :
            n_features : nb of features used (for detection or to be used by the network)
        """
#        assert len(screen_shape) == 3
        self.max_size = max_size
        self.screen_shape = screen_shape
        self.n_variables = n_variables
        self.n_features = n_features
        self.n_goals = n_goals
        self.cursor = 0
        self.full = False
        self.type_network = type_network
        self.screens1 = np.zeros((max_size,) + screen_shape, dtype=np.uint8)
        self.screens2 = np.zeros((max_size,) + screen_shape, dtype=np.uint8)
        if n_variables:
            self.variables = np.zeros((max_size, n_variables), dtype=np.int32)
        if n_features:
            self.features = np.zeros((max_size, n_features), dtype=np.int32)
            self.rewards = np.zeros((max_size, n_features), dtype=np.float32) # for DFP reward = features next state
        else :
            self.rewards = np.zeros(max_size, dtype=np.float32)
        if n_goals :
            self.goals = np.zeros((max_size, n_goals), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.isfinal = np.zeros(max_size, dtype=np.bool)

    @property
    def size(self):
        return self.max_size if self.full else self.cursor

    def add(self, screen1, action, reward, is_final, variables=None, features=None, screen2=None, goals=None):
        assert self.n_variables == 0 or self.n_variables == len(variables)
        assert self.n_features == 0 or self.n_features == len(features)
        self.screens1[self.cursor] = screen1
        if not is_final:
            self.screens2[self.cursor] = screen2
            if self.n_variables:
                self.variables[self.cursor] = variables
            if self.n_features:
                self.features[self.cursor] = features
        if self.n_goals:
            self.goals[self.cursor] = goals
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.isfinal[self.cursor] = is_final
        self.cursor += 1
        if self.cursor >= self.max_size:
            self.cursor = 0
            self.full = True

    def empty(self):
        self.cursor = 0
        self.full = False

    def get_batch(self, batch_size, hist_size):
        """
        Sample a batch of experiences from the replay memory.
        Inputs :
            batch_size : nb of samples needed
            hist_size :  nb of observed frames for s_t (so must be >= 1 for LSTM)
        Returns a dict containing all the raw informations needed to train the network 
        """
        assert self.size > 0, 'replay memory is empty'
        if self.type_network == 'LSTM':
            assert hist_size >= 1, 'history is required for LSTM, not for DFC'
        if self.type_network == 'DFP':
            assert type(hist_size) == list
            l = hist_size
            l.insert(0,0)
            hist_size = l[-1] # simple trick to use a unique code to look for history (both past and future)

        # idx contains the s_t indices
        idx = np.zeros(batch_size, dtype='int32')
        count = 0

        while count < batch_size:

            # index will be the index of s_t
            index = np.random.randint(hist_size - 1, self.size - 1)

            # check that we are not wrapping over the cursor
            if self.cursor <= index + 1 < self.cursor + hist_size:
                continue

            # s_t should not contain any terminal state, so only
            # its last frame (indexed by index) can be final
            if np.any(self.isfinal[index - (hist_size - 1):index]):
                continue

            idx[count] = index
            count += 1
        
        if self.type_network == 'DFP':
            all_indices = idx.reshape((-1, 1)) + 1 - hist_size + np.array(l)
            n_hist = len(l)-1
        else :
            all_indices = idx.reshape((-1, 1)) + np.arange(-(hist_size - 1), 2)
            n_hist = hist_size
            

        screens1 = self.screens1[all_indices]
        screens2 = self.screens2[all_indices]
        variables = self.variables[all_indices] if self.n_variables else None
        features = self.features[all_indices] if self.n_features else None
        goals = self.goals[all_indices] if self.n_goals else None
        actions = self.actions[all_indices[:, :-1]]
        rewards = self.rewards[all_indices[:, :-1]]
        isfinal = self.isfinal[all_indices[:, :-1]]
        
        print('n_hist:{}'.format(n_hist))
        print('r shape: {}'.format(rewards.shape))
        # check batch sizes
        assert idx.shape == (batch_size,)
        assert screens1.shape == (batch_size,n_hist+1) + self.screen_shape
        assert screens2.shape == (batch_size,n_hist+1) + self.screen_shape
        assert (variables is None or variables.shape == (batch_size,
                n_hist + 1, self.n_variables))
        assert (features is None or features.shape == (batch_size,
                n_hist + 1, self.n_features))
        assert (goals is None or goals.shape == (batch_size,
                n_hist + 1, self.n_goals))
        assert actions.shape == (batch_size, n_hist)

        if self.type_network == 'DFP':
            assert rewards.shape == (batch_size, n_hist, 3)
        else:
            assert rewards.shape == (batch_size, n_hist)
        assert isfinal.shape == (batch_size, n_hist)

        return dict(
            screens1=screens1, # np.array batch_size*screen_size*screen_size*hist_size+1
            variables=variables,
            features=features,
            actions=actions,
            rewards=rewards,
            isfinal=isfinal, 
            screens2=screens2,
            goals=goals
        )
