#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:06:17 2019

@author: bastien
"""

from experiment import Experiment
from actions_builder import Action
from keras.optimizers import SGD, Adam, rmsprop
from DFP import DFP_agent
# variables
list_action = ['MOVE_BACKWARD',
                'MOVE_LEFT',
                'SPEED',
                'TURN_LEFT',
                'ATTACK',
                'TURN_RIGHT',
                'MOVE_FORWARD',
                'MOVE_RIGHT',
                'test']
game_features = ['frag_count','health','armor']
game_variables = ['ENNEMY']
scenario = 'basic'
EMBED_GAME_VARIABLES = {
        'ENNEMY' : 0,
        'HEALTH' : 1,
        'WEAPON' : 2,        
        'AMMO' : 3
        }

action_builder = Action(list_action)
e = Experiment(scenario, action_builder, game_features = game_features, visible=True)
#e.start(4, episode_time=9)
#screen, variables, game_features = e.observe_state(game_variables, game_features)


dico_init_network = {}
dico_init_network['image_params'] = {'screen_input_size': (84,84,1)} #dans DFP-Keras
dico_init_network['measure_params'] = {'measure_input_size': 3} 
dico_init_network['goal_params'] = {'goal_input_size': 3 * 6}#i.e. measurement_size * len(timesteps)
dico_init_network['action_params'] = {'nb_actions': len(list_action)}
dico_init_network['expectation_params'] = []
dico_init_network['max_size'] = 20000
dico_init_network['leaky_param'] = 0.2
#learning_rate = 0.00001
dico_init_network['optimizer_params'] = {'type': 'adam'}
dico_init_network['params'] = {'episode_time': 1000, 'frame_skip': 4, 'game_variables': game_variables, 'game_features': game_features}
dico_init_network['variables_names'] = game_variables
dico_init_network['features_names'] = game_features
#dico_init_policy = {}

nb_episodes = 10
map_id = 4

agent_DFP = DFP_agent(dico_init_network)
agent_DFP.train(e, nb_episodes, map_id)




