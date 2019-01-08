#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:06:17 2019

@author: bastien
"""

from experiment import Experiment
from actions_builder import Action

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
game_features = ['frag_count','health']
game_variables = ['ENNEMY']
scenario = 'basic'
action_builder = Action(list_action)
e = Experiment(scenario, action_builder, game_features = game_features, visible=True)
e.start(2)

screen, variables, game_features = e.observe_state(game_variables, game_features)
