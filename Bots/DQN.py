#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:04:05 2019

@author: bastien
"""

from agent import Agent
from keras.layers import Conv2D, Flatten, Dense, Activation
from keras.models import Sequential
import cv2
import numpy as np
import pickle
from utils_network import get_optimizer, saving_stats
from replay_memory import ReplayMemory
from parsers import parse_image_params_dqn

#%% class def

class DQN_agent(Agent):
    """
    DQN agent implementation (for more details, look at )
    """
    
    def __init__(self,
            image_params,
            nb_action,
            logger,
            features = ['health'],
            variables = ['ENNEMY'],
            nb_dense = 128,
            optimizer_params = {'type': 'rmsprop', 'lr': 0.00002, 'clipvalue':1},
            batch_size = 64,
            replay_memory = {'max_size' : 10000, 'screen_shape':(84,84)},
            decrease_eps = lambda epi : 0.05,
            step_btw_train = 64,
            step_btw_save = 2000,
            depth = 4,
            episode_time = 800,
            frame_skip = 4,
            discount_factor = 0.99
                ):
        self.logger = logger
        self.batch_size = batch_size
        self.nb_action = nb_action
        self.replay_memory_p = replay_memory
        self.image_params = image_params
        self.nb_action = nb_action
        self.nb_dense = nb_dense
        self.optimizer_params = optimizer_params
        self.online_network = self.create_network(image_params, nb_dense, nb_action, optimizer_params)
        self.target_network = self.online_network 
        self.decrease_eps = decrease_eps
        self.step_btw_train = step_btw_train
        self.step_btw_save = step_btw_save
        
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
            self.logger.info('input_screen shape is {}'.format(input_screen.shape))
            action = np.random.randint(0,self.nb_action)
            self.logger.info('random action : {}'.format(action))
        else :
            # use trained network to choose action
#            print('using network')
#            print('input dim : {}'.format(input_screen[None,:,:,:].shape))
            pred_q = self.online_network.predict(input_screen[None,:,:,:])
            self.logger.info('q values are : {}'.format(pred_q))
            action = np.argmax(pred_q)
            self.logger.info('opt action : {}'.format(action))
        return action
    
    
    def read_input_state(self,screen, last_states, after=False,MAX_RANGE=255.):
        """
        Use grey level image and specific image definition and stacked frames
        """
        screen_process = screen
        if len(screen.shape) == 3:
            if screen.shape[-1] != 3:
                screen = np.moveaxis(screen,0,-1)
            screen_process = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        input_screen = cv2.resize(screen_process, self.image_size)
        input_screen = input_screen/MAX_RANGE
        screen = np.stack(last_states[-(self.depth-1):]+[input_screen], axis=-1)
        if not after:
            last_states.append(input_screen)
            return screen
        else:
            return input_screen
    
    
    def train(self,map_id, experiment, nb_episodes):
        # variables
        nb_all_steps = 0
        self.list_reward_collected = []
        self.list_reward = []
        self.loss = []
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
            self.logger.info('episode {}'.format(episode))
            
            if episode == 0:
                experiment.new_episode()
            else :
                experiment.reset()
                self.list_reward_collected.append(reward_collected)
                self.logger.info('eps_ellapsed is {}'.format(nb_step))
                print('reward collected is {}'.format(reward_collected))
                self.logger.info('last episode reward collected is {}'.format(reward_collected))
            last_states = []
            reward_collected = 0
            nb_step = 0
            
            # decrease eps according to a fixed policy
            eps = self.decrease_eps(episode)
            self.logger.info('eps for episode {} is {}'.format(episode, eps))
            
            while not experiment.is_final():
#                print(nb_step)
                # get screen and features from the game 
                screen, variables, game_features = experiment.observe_state(self.variables, self.features)
    
                # choose action
                input_screen = self.read_input_state(screen, last_states)
                action  = self.act_opt(eps, input_screen)
                
                # make action and observe resulting state
                r, screen_next, variables_next, game_features_next = experiment.make_action(action, self.variables, self.features, self.frame_skip)
                reward_collected += (self.discount_factor**nb_step)*r
                self.list_reward.append(r)
                if not experiment.is_final():
                    input_screen_next = self.read_input_state(screen, last_states, True)
                else:
                    input_screen_next=None
        
                # save last processed screens / action in the replay memory
                self.replay_mem.add(screen1=last_states[-1],
                                action=action,
                                reward=r,
                                is_final=experiment.is_final(),
                                screen2=input_screen_next
                            )
                
                # train network
                if nb_all_steps >self.depth-1 :
                    loss = self.train_network()
                    self.loss.append(loss)
                
                # change network when needed
                if (nb_all_steps%self.step_btw_train==0) and nb_step >self.depth-1 :
                    print('updating network')
                    self.logger.info('updating network')
                    self.target_network = self.create_network(self.image_params, self.nb_dense, self.nb_action, self.optimizer_params)
                    weight = self.online_network.get_weights()
                    self.target_network.set_weights(weight)
                    
                # count nb of step since start
                nb_step += 1
                nb_all_steps += 1
                
        # save important features on-line
        if (episode%self.step_btw_save==0) and (episode>0):
            print('saving params')
            self.logger.info('saving params')
            saving_stats(episode, experiment.stats, self.online_network, 'DQN_{}'.format(experiment.scenario))
            with open('DQN_list_reward_eps_{}'.format(nb_all_steps)) as fp:
                pickle.dump(self.list_reward_collected,fp)
                    

                
    def test(self,map_id, experiment, nb_episodes):
        """
        Test the trained bot
        """
        # variables
        nb_step = 0
        
        # create game from experiment
        experiment.start(map_id=map_id,
                        episode_time=self.episode_time,
                        log_events=False)
        
        for episode in range(nb_episodes):
            print('episode {}'.format(episode))
            if episode == 0:
                experiment.new_episode()
            else :
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
                
                # make action and observe resulting state
                r, screen_next, variables_next, game_features_next = experiment.make_action(action, self.variables, self.features, self.frame_skip)
        
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
        q2 = np.max(self.target_network.predict(input_screen2), axis=1)
#        print('q2 shape is {}'.format(q2.shape))
        target_q = self.online_network.predict(input_screen1)
#        print('tq shape is {}'.format(target_q.shape))
        target_q[range(target_q.shape[0]), action] = reward + self.discount_factor * (1 - isfinal) * q2
        
        # compute the gradient and update the weights
        loss = self.online_network.train_on_batch(input_screen1, target_q)
        
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
        