#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:04:05 2019

@author: bastien
"""

from agent import Agent
from keras.layers import Input, Conv2D, Flatten, Dense, Activation
from keras.models import Model, Sequential
import cv2
import numpy as np
from utils_network import get_optimizer
from replay_memory import ReplayMemory
from parsers import parse_image_params_dqn

#%% class def

class DQN_agent(Agent):
    """
    DQN agent implementation (for more details, look at )
    """
    
    def __init__(self,
            image_params,
            features = ['frag_count'],
            variables = ['ENNEMY'],
            nb_dense = 128,
            nb_action = 107,
            optimizer_params = {'type': 'adam'},
            batch_size = 4,
            replay_memory = {'max_size' : 10, 'screen_shape':(84,84)},
            decrease_eps = lambda epi : 0.05,
            step_btw_train = 10,
            depth = 4,
            episode_time = 1000,
            frame_skip = 4,
            discount_factor = 0.9
                ):
        self.batch_size = batch_size
        self.nb_action = 5
        self.replay_memory_p = replay_memory
        self.network = self.create_network(image_params, nb_dense, nb_action, optimizer_params)
        self.decrease_eps = decrease_eps
        self.step_btw_train = step_btw_train
        self.features = features
        self.variables = variables
        self.image_size = replay_memory['screen_shape'][:2]
        self.depth = depth
        self.episode_time = episode_time
        self.frame_skip = frame_skip
        self.discount_factor = discount_factor
    
    def act_opt(self,eps, input_screen):
        """
        Choose action according to the eps-greedy policy using the network for inference
        Inputs : 
            eps : eps parameter for the eps-greedy policy
            goal : column vector encoding the goal for each timesteps and each measures
            screen : raw input from the game
            game_features : raw features from the game
        Returns an action coded by an integer
        """        
        # eps-greedy policy used for exploration (if want full exploitation, just set eps to 0)
        if (np.random.rand() < eps) or (input_screen.shape[-1]<4):  # if not enough episode collected, act randomly
            action = np.random.randint(0,self.nb_action)
        else :
            # use trained network to choose action
#            print('using network')
#            print('input dim : {}'.format(input_screen[None,:,:,:].shape))
            pred_q = self.network.predict(input_screen[None,:,:,:])
#            print(pred_q.shape)
            action = np.argmax(pred_q)
        
        return action
    
    
    def read_input_state(self,screen, last_states, after=False):
        """
        Use grey level image and specific image definition and stacked frames
        """
        if screen.shape[-1] != 3:
            screen = np.moveaxis(screen,0,-1)
        input_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        input_screen = cv2.resize(input_screen, self.image_size)
        screen = np.stack(last_states[-(self.depth-1):]+[input_screen], axis=-1)
        if not after:
            last_states.append(input_screen)
            return screen
        else:
            return input_screen
    
    
    def train(self,map_id, experiment, nb_episodes):
        # variables
        nb_step = 0
        
        # create game from experiment
        experiment.start(map_id=map_id,
                        episode_time=self.episode_time,
                        log_events=False)
        
        # create replay memory
        self.replay_mem = ReplayMemory(self.replay_memory_p['max_size'],
                                       self.replay_memory_p['screen_shape'],
                                       type_network='DQN')
        
        # run the game
        for episode in range(nb_episodes):
            print('episode {}'.format(episode))
            experiment.reset()
            last_states = []
            
            while not experiment.is_final():
#                print(nb_step)
                # get screen and features from the game 
                screen, variables, game_features = experiment.observe_state(self.variables, self.features)
    
                # decrease eps according to a fixed policy
                eps = self.decrease_eps(episode)
                
                # choose action
                input_screen = self.read_input_state(screen, last_states)
#                print(input_screen.shape)
                action  = self.act_opt(eps, input_screen)
                
                # make action and observe resulting state (plays the role of the reward)
                r = experiment.make_action(action, self.frame_skip)
#                print('reward is {}'.format(r))
                if not experiment.is_final():
                    screen_next, variables_next, game_features_next = experiment.observe_state(self.variables, self.features)
                    input_screen_next = self.read_input_state(screen, last_states, True)
                else:
                    input_screen_next=None
                # save last processed screens / features / action in the replay memory
                self.replay_mem.add( screen1=last_states[-1],
                                action=action,
                                reward=r,
                                is_final=experiment.is_final(),
                                screen2=input_screen_next
                            )
                
                # train network if needed
                if (nb_step%self.step_btw_train==0) and nb_step !=0 :
                    print('training')
                    self.train_network()
                
                
                # count nb of step since start
                nb_step += 1
        
        
    def train_network(self):
        """
        Sample from the replay memory and trained the network with a simple batch on 
        these samples
        """
        batch = self.replay_mem.get_batch(self.batch_size, 3)
        input_screen1 = np.moveaxis(batch['screens1'],1,-1)
        input_screen2 = np.moveaxis(batch['screens2'],1,-1)
        reward = batch['rewards'][:,-1]
        isfinal = batch['isfinal'][:,-1]
        action = batch['actions'][:,-1]
        
        # compute target values
        q2 = np.max(self.network.predict(input_screen2), axis=1)
#        print('q2 shape is {}'.format(q2.shape))
        target_q = self.network.predict(input_screen1)
#        print('tq shape is {}'.format(target_q.shape))
        target_q[range(target_q.shape[0]), action] = reward + self.discount_factor * (1 - isfinal) * q2
        
        # compute the gradient and update the weights
        loss = self.network.train_on_batch(input_screen1, target_q)
        
        return loss
    
    
    @staticmethod
    def create_network(image_params, nb_dense, nb_actions, optimizer_params):
        """
        create DQN network as described in paper from Mnih & al
        """
        # parse network inputs parameters
        screen_input_size, s1, s2, s3 = parse_image_params_dqn(image_params)
        
        # Define optimizer
        optimizer = get_optimizer(optimizer_params)
        
        # build network
        model = Sequential()
        model.add(Conv2D(s1['channel'], (s1['kernel'],s1['kernel']), strides=(s1['stride'], s1['stride'])
                            ,input_shape=screen_input_size))  #84*84*4
        model.add(Activation('relu'))
        model.add(Conv2D(s2['channel'], (s2['kernel'],s2['kernel']), strides=(s2['stride'], s2['stride'])))
        model.add(Activation('relu'))
        model.add(Conv2D(s3['channel'], (s3['kernel'],s3['kernel']), strides=(s3['stride'], s3['stride'])))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(nb_dense))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        
        # compile model
        model.compile(loss='mse', optimizer=optimizer)
        
        return model
    
    
    
    
#%% Methods
        