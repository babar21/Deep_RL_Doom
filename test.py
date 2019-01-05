#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:37:20 2019

@author: bastien
"""

image_params = dict()
image_params['screen_input_size'] = 128

measure_params=dict()
measure_params['measure_input_size']=3

goal_params=dict()
goal_params['goal_input_size']=18

action_params=dict()
action_params['nb_actions']=256
action_params['a1']= {'output':1024}

expectation_params=dict()
expectation_params['e1']= {'output':1024}

optimizer_params = dict()
optimizer_params['type'] = 'adam'

leaky_param = 0.2

