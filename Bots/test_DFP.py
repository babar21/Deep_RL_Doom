#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:55:05 2019

@author: bastien
"""

#%% Import
from game_motor.experiment import Experiment, process_game_statistics
from game_motor.actions_builder import Action
from game_motor.reward import Reward
from DFP import DFP_agent
import matplotlib.pyplot as plt
import pickle
import numpy as np
from parsers import *

#%% Logger
import logging
from logging import FileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
file_handler = FileHandler('activity_dfp.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#%% Define the environment
list_action =[
		'MOVE_LEFT', 
		'MOVE_RIGHT', 
		'ATTACK']

screen_resolution = 'RES_640X480'
living_reward = -1

action_builder = Action(list_action) # create actions

reward_builder = Reward() # default rewards
custom_reward=False

game_features = ['frag_count','health']
game_variables = ['HEALTH']

scenario = 'simpler_basic'

experiment = Experiment(scenario, action_builder,reward_builder, logger,
               custom_reward=custom_reward,living_reward=living_reward, game_features = game_features, visible=False)

#%% Define the agent
n_action = action_builder.n_actions
screen_shape = (84,84)
image_params = {'screen_input_size' : screen_shape + (1,)}
measure_params = {'measure_input_size' : 3}
goal_params = {'goal_input_size' : 3*6}
action_params = {'nb_actions' : n_action}
expectation_params = {}

#decrease_eps = lambda step : 0.02 + 145000. / (float(step) + 150000.)

def exploration_rate(epoch, nb_episode):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * nb_episode # 10% of learning time
        eps_decay_epochs = 0.6 * nb_episode  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps + (epoch - const_eps_epochs)/(const_eps_epochs - 
                                           eps_decay_epochs) * (start_eps - end_eps)
        else:
            return end_eps


map_id = 1
nb_episodes = 200
nb_episodes_test = 5
decrease_eps = lambda eps : exploration_rate(eps, nb_episodes)



agent = DFP_agent(image_params,
                 measure_params, 
                 goal_params, 
                 expectation_params, 
                 action_params,
                 n_action,
                 logger, 
                 decrease_eps=decrease_eps)

#%% Run the training, then testing

agent.train(experiment, nb_episodes, map_id)








