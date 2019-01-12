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
from replay_memory import ReplayMemory

import cv2


class DFP_agent(Agent):
    """
    DFP agent implementation (for more details, look at https://arxiv.org/abs/1611.01779)
    Subclass of Abstract class Agent
    """
    def __init__(self, 
                 image_params,
                 measure_params, 
                 goal_params, 
                 expectation_params, 
                 action_params,
                 nb_action,
                 logger,
                 goal_mode = 'fixed',
                 optimizer_params = {'type' : 'adam'}, 
                 leaky_param = 0.2,
                 features = ['frag_count', 'health', 'sel_ammo'],
                 variables = ['ENNEMY'],
                 replay_memory = {'max_size' : 10000, 'screen_shape':(84,84)},
                 decrease_eps = lambda epi : 0.1,
                 step_btw_train = 2,
                 episode_time = 300,
                 frame_skip = 4,
                 batch_size = 64,
                 time_steps = [1,2,4,8,16,32],
                 ):
        """
        Read bot parameters from different dicts and initialize the bot
        Inputs :
            dico_init_network
            dico_init_policy
        """
        #Initialize params
        self.batch_size = batch_size
        self.step_btw_train = step_btw_train
        self.time_steps = time_steps
        self.nb_action = nb_action
        self.episode_time = episode_time
        self.frame_skip = frame_skip
        self.goal_mode = goal_mode
        
        self.logger = logger
        self.replay_memory_p = replay_memory
        self.variables = variables     
        self.features = features
        self.n_features = len(self.features)
        self.n_goals = len(self.features)*len(self.time_steps)
        self.n_variables = len(self.variables)
        self.replay_memory = {'screen_shape': replay_memory['screen_shape'], 'n_variables': self.n_variables, 'n_features': self.n_features}
        self.image_size = self.replay_memory['screen_shape'][:2]

        # init network
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
            self.logger.info('random action : {}'.format(action))
        else :
            # use trained network to choose action
            pred_measure = self.network.predict([input_screen[None,:,:,:], input_game_features, goal])
            pred_measure_calc = np.reshape(pred_measure, (self.nb_action, len(goal)))
            list_act = np.dot(pred_measure_calc,goal)
            action = np.argmax(list_act)
            self.logger.info('opt action : {}'.format(action))
        return action


    def read_input_state(self, screen, game_features, last_states, after=False, MAX_RANGE=255.):
        """
        Use grey level image and specific image definition
        """
        if screen.shape[-1] != 3:
            screen = np.moveaxis(screen,0,-1)

        input_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        input_screen = cv2.resize(input_screen, self.image_size)
        input_screen = input_screen/MAX_RANGE
        input_game_features = np.zeros(self.n_features)
        i=0
        for features in self.features:
            input_game_features[i] = game_features[features]
            i +=1
        
        if not after:
            last_states.append(input_screen)
            return screen, input_game_features
        
        else:
            return input_screen, input_game_features
    
    
    def train(self, experiment, nb_episodes, map_id):
        """
        Train the bot according to an eps-greedy policy
        Use a replay memory (see dedicated class)
        Inputs :
            experiment : object from the experiment class, which contains the game motor
        """
        nb_all_steps = 0
        
        # create game from experiment
        experiment.start(map_id=map_id,
                         episode_time=self.episode_time,
                         log_events=False)
        
        # create replay memory
        self.replay_mem = ReplayMemory(self.replay_memory_p['max_size'], 
                                       self.replay_memory_p['screen_shape'], 
                                       type_network='DFP', n_features=self.n_features, 
                                       n_goals=self.n_goals)
        
        # run training
        for episode in range(nb_episodes):
            print('episode {}'.format(episode))
            self.logger.info('episode {}'.format(episode))
            
            # initialize goal for each episode
            assert self.goal_mode in ['fixed', 'random_1', 'random_2']
            if self.goal_mode == 'fixed':
                goal = np.array([0.5, 0.5, 1])
            if self.goal_mode == 'random_1':
                goal = np.random.uniform(0, 1, size=3)
            if self.goal_mode == 'random_2':
                goal = np.random.uniform(-1, 1, size=3)
            goal = np.tile(goal, len(self.time_steps))
            
            if episode == 0:
                experiment.new_episode()
            else :
                experiment.reset()
            
            # variables 
            last_states = []
            nb_step = 0
            
            # decrease eps according to a fixed policy
            eps = self.decrease_eps(episode)
            self.logger.info('eps for episode {} is {}'.format(episode, eps))
            
            while not experiment.is_final():
                # get screen and features from the game 
                screen, game_variables, game_features = experiment.observe_state(self.variables, self.features)
                # choose action
                input_screen, input_game_features = self.read_input_state(screen, game_features, last_states)
                action = self.act_opt(eps, input_screen, input_game_features, goal)
                
                # make action and observe resulting measurement (plays the role of the reward)
                r, screen_next, variables_next, game_features_next = experiment.make_action(action, self.variables, self.features, self.frame_skip)
                
                # calculate reward based on goal an
                if not experiment.is_final():
                    input_screen_next, input_game_features_next = self.read_input_state(screen, game_features, last_states, True)
                else:
                    input_screen_next=None
 
                self.replay_mem.add(screen1=last_states[-1],
                                    action=action,
                                    reward=r,
                                    features=input_game_features,
                                    is_final=experiment.is_final(),
                                    screen2=input_screen_next,
                                    goals=goal)

                # train network if needed
                if (nb_step%self.step_btw_train==0) and (nb_all_steps > self.time_steps[-1]) :
                    print('updating network')
                    self.logger.info('updating network')
                    self.train_network(self.replay_mem)

                # count nb of steps since start
                nb_step += 1
                nb_all_steps += 1
                
        
    def train_network(self, replay_memory):
        """
        train the network according to a batch size and a replay memory
        """
        # Load a batch from replay memory
        hist_size = self.time_steps # TODO mettre en dur sinon pb
        print(hist_size)
        batch = replay_memory.get_batch(self.batch_size, hist_size)        
        
        # Store the training input
        input_screen1 = batch['screens1'][:,0,:,:]
        action = batch['actions'][:,0]
        current_features = batch['features'][:,0,:]
        future_features = batch['features'][:,1:,:]
        future_features = np.reshape(future_features, (future_features.shape[0],
                                                       future_features.shape[1]*future_features.shape[2]))
        current_goal = batch['goals'][:,0,:]
        
        print('coucou')
        # Predict features target
        feature_target = self.network.predict([input_screen1[:,:,:,None], current_features, current_goal]) # flatten vector nb_actions * len(goa)
        feature_target_reshape = np.reshape(feature_target, (feature_target.shape[0], self.nb_action, self.n_goals))

        # change value to predict with observed features
        feature_target_reshape[range(feature_target_reshape.shape[0]),action,:] = future_features
        f_target = np.reshape(feature_target_reshape,(feature_target.shape[0], self.nb_action*self.n_goals)) 
        
        # compute the gradient and update the weights
        loss = self.network.train_on_batch([input_screen1[:,:,:,None], current_features, current_goal], f_target)

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