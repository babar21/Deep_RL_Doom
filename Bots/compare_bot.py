#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 22:00:16 2019

@author: bastien
"""

#%% Import
from game_motor.experiment import Experiment, process_game_statistics
from game_motor.actions_builder import Action
from game_motor.reward import Reward
from DFP import DFP_agent

import matplotlib.pyplot as plt
import pickle
import keras
import numpy as np
from parsers import *


#%% Loggers
import logging
from logging import FileHandler

logger1 = logging.getLogger()
logger1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
file_handler1 = FileHandler('basic_1_naive_test.log')
file_handler1.setLevel(logging.DEBUG)
file_handler1.setFormatter(formatter)
logger1.addHandler(file_handler1)

logger2 = logging.getLogger()
logger2.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
file_handler2 = FileHandler('basic_1_trained_test.log')
file_handler2.setLevel(logging.DEBUG)
file_handler2.setFormatter(formatter)
logger2.addHandler(file_handler2)


#%% Define the environment
list_action =[
		'TURN_LEFT',
		'TURN_RIGHT', 
		'MOVE_FORWARD']

screen_resolution = 'RES_160X120'
screen_format='GRAY8'
living_reward = 1

action_builder = Action(list_action) # create actions

reward_builder = Reward() # default rewards
custom_reward=False

game_features = ['health']
game_variables = ['HEALTH']

scenario = 'D1_basic'

experiment = Experiment(scenario, action_builder,reward_builder, logger1,
               custom_reward=custom_reward,living_reward=living_reward, 
               game_features = game_features, visible=False,
               screen_format = screen_format)


#%% Define the agent
# game params
n_action = action_builder.n_actions
screen_shape = (84,84)

# goal params
features = ['health']
rel_weight = [1.]

# network params
image_params = {'screen_input_size' : screen_shape + (1,)}
measure_params = {'measure_input_size' : len(features)}
goal_params = {'goal_input_size' : len(features)*6} # only health as measurement 
action_params = {'nb_actions' : n_action}
expectation_params = {}

map_id = 1
nb_episodes = 100
#decrease_eps = lambda eps : exploration_rate(eps, nb_episodes)
decrease_eps_trained = lambda step : 0.05
decrease_eps_naive = lambda step : 1

# agents definition 
# naive
agent_naive = DFP_agent(image_params,
                 measure_params, 
                 goal_params, 
                 expectation_params, 
                 action_params,
                 n_action,
                 logger2, 
                 decrease_eps=decrease_eps_naive,
                 features = features,
                 rel_weight = rel_weight,
                 step_btw_train = np.inf
                 )
# trained
agent_trained = DFP_agent(image_params,
                 measure_params, 
                 goal_params, 
                 expectation_params, 
                 action_params,
                 n_action,
                 logger1, 
                 decrease_eps=decrease_eps_trained,
                 features = features,
                 rel_weight = rel_weight, 
                 step_btw_train = np.inf
                 )


n = keras.models.load_model('/Users/bastien/Documents/ENS_2018-2019/Reinforcement_Learning/RL_Doom/experiments/DFP_D1_basic_network_38000.dms')
agent_trained.network = n


# test naive
#agent_naive.train(experiment, nb_episodes, map_id)
#stats_naive = experiment.stats
#with open('compare_bots_naive','wb') as fp:
#    pickle.dump(stats_naive,fp)

# test trained 
experiment.stats[map_id]= []
agent_trained.train(experiment, nb_episodes, map_id)
stats_trained = experiment.stats
with open('compare_bots_trained_38000','wb') as fp:
    pickle.dump(stats_trained,fp)





