#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:54:59 2019

@author: bastien
"""

from agent import Agent
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Lambda, Reshape, LeakyReLU, Add
from keras.models import Model
import numpy as np
from utils_network import normalize_layer, get_optimizer
from parsers import parse_image_params, parse_measure_params, parse_goal_params, parse_action_params, parse_expectation_params
from replay_memory import ReplayMemory
import cv2


class DFP_agent(Agent):
    """
    DFP agent implementation (for more details, look at https://arxiv.org/abs/1611.01779)
    Subclass of Abstract class Agent
    """
    def __init__(self, dico_init_network, dico_init_policy):
        """
        Read bot parameters from different dicts and initialize the bot
        Inputs :
            dico_init_network
            dico_init_policy
        """
        self.
        self.step_btw_train = 64 
        self.decrease_eps = 'function to decrease eps'
        self.max_size = dico_init_network['max_size']
        self.nb_action = ''
        self.network = create_network(image_params, measure_params, goal_params, expectation_params, 
                               action_params, optimizer_params, leaky_param)

    
    def act_opt(self, eps, screen, game_features, goal):
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
        if np.random.rand() < eps :
            action = np.random.randint(0,self.nb_action)
        else :
            # use trained network to choose action
            pred_measure = self.network.predict([input_screen, input_game_features, goal])
            pred_measure_calc = np.reshape(pred_measure, (nb_actions, len(goal_calc)))
            list_act = np.dot(pred_measure_calc,goal_calc)
            action = np.argmax(list_act)
        
        return action
    
    
    def read_input_state(self, screen, game_features):
        """
        Use grey level image and specific image definition
        """
        input_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        input_screen = cv2.resize(input_screen, (self.image_size, self.image_size))
        input_game_features = np.zeros(len(self.game_features))
        i=0
        for features in self.game_features :
            input_game_features[i] = game_features[features]
            i +=1
        return input_screen, input_game_features
    
    
    def train(self, experiment, nb_episodes):
        """
        Train the bot according to an eps-greedy policy
        Use a replay memory (see dedicated class)
        Inputs : 
            experiment : object from the experiment class, which contains the game motor
        """
        # variables
        last_states = []
        nb_step = 0
        
        # create game from experiment
        experiment.start(map_id=map_id,
                        episode_time=self.params.episode_time,
                        log_events=False)
        
        # create replay memory
        replay_mem = ReplayMemory(self.max_size, self.replay_memory['screen_shape'], self.replay_memory['n_variables'], self.replay_memory['n_features'])
        
        # run training
        for episode in range(nb_episodes):
            
            # reset game for a new episode
            experiment.reset()   
            
            while not experiment.is_final():
                # get screen and features from the game 
                screen, game_features = experiment.observe_state(params, last_states)
                
                # decrease eps according to a fixed policy
                eps = self.decrease_eps(self,episode)
                
                # choose action
                input_screen, input_game_features = self.read_input_state(screen, game_features)
                action,  = self.act_opt(eps, input_screen, input_game_features, goal)
                
                # make action and observe resulting measurement (plays the role of the reward)
                experiment.make_action(action, self.params.frame_skip)
                screen, game_features = experiment.observe_state(params, last_states)
                input_screen_next, input_game_features_next = self.read_input_state(screen, game_features)
                
                # save last processed screens / features / action in the replay memory
                replay_mem.add( screen=input_screen,
                                variables=None,
                                features=input_game_features,
                                action=action,
                                reward=input_game_features_next,
                                is_final=self.game.is_final()
                            )
                
                # train network if needed
                if nb_step%self.step_btw_train :
                    self.train(replay_mem)
                
                
                # count nb of step since start
                nb_step += nb_step
      
        
    def train_network(self, replay_memory):
        """
        train the network according to a batch size and a replay memory
        """
        train_set = replay_memory.get_batch(self.batch_size, 1)
        
        
        
        
    
    
    @staticmethod    
    def create_network(image_params, measure_params, goal_params, expectation_params, 
                               action_params, optimizer_params, leaky_param, norm=True, split=True):
        """
        Create the neural network proposed in the paper
        Inputs: 
            image_params : dict with keys
            measure_params = dict with keys
            goal_params = dict with keys
            norm : to add normalization step
            split :  to add expectation stream
        Returns a flatten tensor with dims (nb_actions*goal_input_size) obtained with Flatten
        """
        # check network parameters
        screen_input_size, s1, s2, s3, s4 = parse_image_params(image_params)
        measure_input_size, m1, m2, m3 = parse_measure_params(measure_params)
        goal_input_size, g1, g2, g3 = parse_goal_params(goal_params)
        nb_actions, a1 = parse_action_params(action_params)
        e1 = parse_expectation_params(expectation_params)
        
        # Define optimizer
        optimizer = get_optimizer(optimizer_params)
        
        # Image stream
        screen_input = Input(shape=screen_input_size) 
        s1 = Conv2D(s1['channel'], (s1['kernel'],s1['kernel']), strides=(s1['stride'], s1['stride']), activation='linear')(screen_input)
        s1 = LeakyReLU(alpha=leaky_param)(s1)
        s2 = Conv2D(s2['channel'], (s2['kernel'],s2['kernel']), strides=(s2['stride'], s2['stride']), activation='linear')(s1)
        s2 = LeakyReLU(alpha=leaky_param)(s2)
        s3 = Conv2D(s3['channel'], (s3['kernel'],s3['kernel']), strides=(s3['stride'], s3['stride']), activation='linear')(s2)
        s3 = LeakyReLU(alpha=leaky_param)(s3)
        sf = Flatten()(s3)
        s4 = Dense(s4['output'],activation='linear')(sf)
        s4 = LeakyReLU(alpha=leaky_param)(s4)
        
        # Measurement stream
        measure_input = Input(shape=(measure_input_size,))
        m1 = Dense(m1['output'], activation='linear')(measure_input)
        m1 = LeakyReLU(alpha=leaky_param)(m1)
        m2 = Dense(m2['output'], activation='linear')(m1)
        m2 = LeakyReLU(alpha=leaky_param)(m2)
        m3 = Dense(m3['output'], activation='linear')(m2)
        m3 = LeakyReLU(alpha=leaky_param)(m3)
        
        # Goal stream
        goal_input = Input(shape=(goal_input_size,))
        g1 = Dense(g1['output'],activation='linear')(goal_input)
        g1 = LeakyReLU(alpha=leaky_param)(g1)
        g2 = Dense(g2['output'],activation='linear')(g1)
        g2 = LeakyReLU(alpha=leaky_param)(g2)
        g3 = Dense(g3['output'],activation='linear')(g2)
        g3 = LeakyReLU(alpha=leaky_param)(g3)
        
        # Concatenate (image,measure,goal)
        concat = Concatenate()([s4,m3,g3])
        
        # Action stream with normalisation or not
        a1 = Dense(a1['output'], activation='linear')(concat)
        a1 = LeakyReLU(alpha=leaky_param)(a1)
        pred = Dense(goal_input_size*nb_actions, activation='linear')(a1)
        pred = LeakyReLU(alpha=leaky_param)(pred)
        pred = Reshape((nb_actions,goal_input_size))(pred)
        if norm==True :
            pred = Lambda(normalize_layer)(pred)
        
        if split==True :
            # Expectation stream
            e1 = Dense(e1['output'], activation='linear')(concat)
            e1 = LeakyReLU(alpha=leaky_param)(e1)
            e2 = Dense(goal_input_size, activation='linear')(e1)
            e2 = LeakyReLU(alpha=leaky_param)(e2)
            pred = Add()([e2,pred])
        
        pred = Flatten()(pred)
        
        # Final model
        model = Model(inputs=[screen_input, measure_input, goal_input], 
                                      outputs=pred)
        
        # compile model
        model.compile(loss='mse', optimizer=optimizer)
        
        return model

        
        
        
        