#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for vanilla DQN
"""
#%% Import
from game_motor.experiment import Experiment, process_game_statistics
from game_motor.actions_builder import Action
from game_motor.reward import Reward
from DQN import DQN_agent
import matplotlib.pyplot as plt
import pickle

#%% Logger
import logging
from logging import FileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
file_handler = FileHandler('activity.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

experiment = Experiment(scenario, action_builder,reward_builder, logger,
               custom_reward=custom_reward,living_reward=living_reward, 
               game_features = game_features, visible=False,
               screen_format = screen_format)

#%% Define the agent
screen_shape = (84,84)
depth = 4
image_params = {'screen_input_size' : screen_shape + (depth,)}
n_actions = action_builder.n_actions
#decrease_eps = lambda step : 0.02 + 145000. / (float(step) + 150000.)

#def exploration_rate(epoch, nb_episode):
#        """# Define exploration rate change over time"""
#        start_eps = 1.0
#        end_eps = 0.1
#        const_eps_epochs = 0.1 * nb_episode # 10% of learning time
#        eps_decay_epochs = 0.6 * nb_episode  # 60% of learning time
#
#        if epoch < const_eps_epochs:
#            return start_eps
#        elif epoch < eps_decay_epochs:
#            # Linear decay
#            return start_eps + (epoch - const_eps_epochs)/(const_eps_epochs - 
#                                           eps_decay_epochs) * (start_eps - end_eps)
#        else:
#            return end_eps


map_id = 1
nb_episodes = 800000
nb_episodes_test = 5
#decrease_eps = lambda eps : exploration_rate(eps, nb_episodes)
decrease_eps = lambda step : 0.02 + 145000. / (float(step) + 150000.)

agent = DQN_agent(image_params, n_actions, logger, decrease_eps=decrease_eps)

#%% Run the training, then testing
agent.train(map_id, experiment, nb_episodes)

#agent.decrease_eps = lambda eps : 0
##agent.test(map_id, experiment, nb_episodes_test)
#
#r = process_game_statistics(experiment.stats)
#with open('dqn_stats', 'wb') as fp:
#    pickle.dump(r)
#
#list_collected_reward = agent.list_reward_collected
#with open('dqn_rewards', 'wb') as fp:
#    pickle.dump(list_collected_reward)

#list_loss =  agent.list_loss
#with open('dqn_loss', 'wb') as fp:
#    pickle.dump(list_loss)
