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
#from replay_memory import ReplayMemory
from replay_memory_2 import ReplayMemory

import cv2


class DFP_agent(Agent):
    """
    DFP agent implementation (for more details, look at https://arxiv.org/abs/1611.01779)
    Subclass of Abstract class Agent
    """
#    def __init__(self, dico_init_network, dico_init_policy):
    def __init__(self, dico_init_network):
        """
        Read bot parameters from different dicts and initialize the bot
        Inputs :
            dico_init_network
            dico_init_policy
        """
        #Initialize params
        self.batch_size = 32
        self.step_btw_train = 64 
        self.time_steps = [1,2,4,8,16,32]
        self.max_size = dico_init_network['max_size']
        self.nb_action = dico_init_network['action_params']['nb_actions']
        self.episode_time = dico_init_network['params']['episode_time']
        self.frame_skip = dico_init_network['params']['frame_skip']
        
        image_params = dico_init_network['image_params']
        measure_params = dico_init_network['measure_params']
        goal_params = dico_init_network['goal_params']
        expectation_params = dico_init_network['expectation_params']
        action_params = dico_init_network['action_params']
        leaky_param = dico_init_network['leaky_param']
        optimizer_params = dico_init_network['optimizer_params']
    
#        self.replay_memory = {'screen_shape': (84,84,1), 'n_variables': 8, 'n_features': 3}
        n_variables = len(dico_init_network['variables_names'])
        self.replay_memory = {'screen_shape': (84,84,4), 'n_variables': n_variables, 'n_features': 3}
        self.image_size = self.replay_memory['screen_shape'][:2]
        self.variables_names = dico_init_network['variables_names']        
        self.features_names = dico_init_network['features_names']
       
        self.network = self.create_network(image_params, measure_params, goal_params, expectation_params, 
                               action_params, optimizer_params, leaky_param)


    def act_opt(self, eps, input_screen, input_game_features, goal):
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
        if (np.random.rand() < eps) or (input_screen.shape[-1]<4):
            action = np.random.randint(0,self.nb_action)
        else :
            # use trained network to choose action
            pred_measure = self.network.predict([input_screen[None,:,:,:], input_game_features, goal])
            pred_measure_calc = np.reshape(pred_measure, (self.nb_action, len(goal)))
#            pred_measure_calc = np.reshape(pred_measure, (self.nb_action, len(goal_calc)))
#            list_act = np.dot(pred_measure_calc,goal_calc)
            list_act = np.dot(pred_measure_calc,goal)
            action = np.argmax(list_act)
        
        return action

#    def act_opt(self,eps, input_screen):
#        """
#        Choose action according to the eps-greedy policy using the network for inference
#        Inputs : 
#            eps : eps parameter for the eps-greedy policy
#            goal : column vector encoding the goal for each timesteps and each measures
#            screen : raw input from the game
#            game_features : raw features from the game
#        Returns an action coded by an integer
#        """        
#        # eps-greedy policy used for exploration (if want full exploitation, just set eps to 0)
#        if (np.random.rand() < eps) or (input_screen.shape[-1]<4):  # if not enough episode collected, act randomly
#            action = np.random.randint(0,self.nb_action)
#        else :
#            # use trained network to choose action
##            print('using network')
##            print('input dim : {}'.format(input_screen[None,:,:,:].shape))
#            pred_q = self.network.predict(input_screen[None,:,:,:])
##            print(pred_q.shape)
#            action = np.argmax(pred_q)
#        
#        return action
    
    def read_input_state(self, screen, last_states, game_features, after=False):
        """
        Use grey level image and specific image definition
        """
        if screen.shape[-1] != 3:
            screen = np.moveaxis(screen,0,-1)

        input_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        input_screen = cv2.resize(input_screen, self.image_size)
        input_game_features = np.zeros(len(game_features))
        i=0
        for features in game_features :
            input_game_features[i] = game_features[features]
            i +=1
        
        if not after:
            last_states.append(input_screen)
            return screen, input_game_features
        
        else:
            return input_screen, input_game_features
    
    
    def train(self, experiment, nb_episodes, map_id, goal_mode='fixed'):
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
                        episode_time=self.episode_time,
                        log_events=False)
        
        # create replay memory
#        replay_mem = ReplayMemory(self.max_size, self.replay_memory['screen_shape'], self.replay_memory['n_variables'], 
#                                  self.replay_memory['n_features'], is_DFP=True)
        self.replay_mem = ReplayMemory(self.max_size, self.replay_memory['screen_shape'][:2], type_network='DFP', n_variables=self.replay_memory['n_variables'], 
                                  n_features=self.replay_memory['n_features'], n_goals=3)
        
        # run training
        for episode in range(nb_episodes):
            
            #Initialize goal for each episode
            assert goal_mode in ['fixed', 'random_1', 'random_2']
            if goal_mode == 'fixed':
                goal = np.array([0.5, 0.5, 1])
            if goal_mode == 'random_1':
                goal = np.random.uniform(0, 1, size=3)
            if goal_mode == 'random_2':
                goal = np.random.uniform(-1, 1, size=3)

            # reset game for a new episode
            experiment.reset()   
            
            while not experiment.is_final():
#                print(experiment.is_final())
                # get screen and features from the game 
#                screen, game_features = experiment.observe_state(self.params, last_states)
                screen, game_variables, game_features = experiment.observe_state(self.variables_names, self.features_names)
                
                # at each step, decrease eps according to a fixed policy
                eps = self.decrease_eps(nb_step)
                
                # choose action
                input_screen, input_game_features = self.read_input_state(screen, last_states, game_features)
#                print(input_screen.shape)
#                print(input_game_features.shape)
                action = self.act_opt(eps, input_screen, input_game_features, goal)
                
                # make action and observe resulting measurement (plays the role of the reward)
                r = experiment.make_action(action, self.frame_skip)
#                screen, game_features = experiment.observe_state(self.params, last_states)
                if not experiment.is_final():
                    screen_next, game_variables_next, game_features_next = experiment.observe_state(self.variables_names, self.features_names)
                    input_screen_next, input_game_features_next = self.read_input_state(screen, last_states, game_features, after=True)
  
                else:
                    input_screen_next = None
                    input_game_features_next = None
 
                self.replay_mem.add( screen1=last_states[-1],
                                action=action,
                                reward=r,
                                features=input_game_features_next,
                                variables = [v for v in game_variables.values()],
                                is_final=experiment.is_final(),
                                screen2=input_screen_next,
                                goals=goal
)

                
                # save last processed screens / features / action in the replay memory

##replay_mem_v1
#                replay_mem.add( screen=screen,
#                                variables=[v for v in game_variables.values()],
#                                features=input_game_features,
#                                action=action,
#                                reward=input_game_features_next,
#                                is_final=experiment.is_final(), 
#                                goal=goal
#                            )

   
#                print(nb_step)
                # train network if needed
                if nb_step >=10:
                    if nb_step%self.step_btw_train==0 :
                        self.train_network(self.replay_mem)

                # count nb of steps since start
                nb_step += 1      
        
    def train_network(self, replay_memory):
        """
        train the network according to a batch size and a replay memory
        """
        print('entering train_network')
        #Load a batch from replay memory
        hist_size = self.time_steps
        batch = replay_memory.get_batch(self.batch_size, hist_size)        
        print('got batch')
        #Store the training input
        #future_features = train_set['future_features']
        input_goals = np.zeros((self.batch_size, 18))
        input_features = np.zeros((self.batch_size, 3))
        future_features = np.zeros((self.batch_size, len(hist_size)-1, 3))
#        print(len(hist_size))
#        print(train_set['features'][0][1:].shape)
        for i in range(self.batch_size):
            input_goals[i] = batch['goals'][i][1:].flatten()
            
            future_features[i] = batch['features'][i][1:]
            input_features[i] = batch['features'][i][0]
        print('got future features')
        
        input_screen1 = np.moveaxis(batch['screens1'],1,-1)
        input_screen2 = np.moveaxis(batch['screens2'],1,-1)
        #Predict target
        print(input_screen1.shape)
        print(input_screen2.shape)
        print(future_features.shape)
        print(batch['goals'][0][1:].shape)
        print(input_goals[0].shape)
        f_target = self.network.predict([input_screen1[:,:,:,0].reshape(32, 84, 84, 1), input_features, input_goals])
        print(f_target.shape)
        print(batch['actions'][0])
        for i in range(self.batch_size):
#            f_target[train_set['actions'][i]][i,:] = future_features[i]
            f_target[batch['actions'][i]][i,:] = future_features[i]
        
        loss = self.network.train_on_batch([input_screen1[:,:,:,0].reshape(32, 84, 84, 1), input_features, input_goals], f_target)

#        input_screen1 = np.moveaxis(batch['screens1'],1,-1)
#        input_screen2 = np.moveaxis(batch['screens2'],1,-1)
#        reward = batch['rewards'][:,-1]
#        isfinal = batch['isfinal'][:,-1]
#        action = batch['actions'][:,-1]
#        
#        # compute target values
#        q2 = np.max(self.network.predict(input_screen2), axis=1)
##        print('q2 shape is {}'.format(q2.shape))
#        target_q = self.network.predict(input_screen1)
##        print('tq shape is {}'.format(target_q.shape))
#        target_q[range(target_q.shape[0]), action] = reward + self.discount_factor * (1 - isfinal) * q2
#        
#        # compute the gradient and update the weights
#        loss = self.network.train_on_batch(input_screen1, target_q)        

        return loss
        
    def decrease_eps(self, step):
         return (0.02 + 145000. / (float(step) + 150000.))       
        

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