#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:30:35 2019

@author: bastien
"""
s1 = dict()
s2 = dict()
s3 = dict()
s1['channel']=32
s1['kernel']=8
s1['stride']=4

s2['channel']=64
s2['kernel']=4
s2['stride']=2
s3['channel']=64
s3['kernel']=3
s3['stride']=1

image_params = {'screen_input_size' : (84,84,4),
                's1' : s1,
                's2' : s2,
                's3' : s3
        }


from DQN import DQN_agent
from replay_memory import ReplayMemory

a = DQN_agent(image_params)
#replay_mem = ReplayMemory(a.replay_memory['max_size'], 
#                          a.replay_memory['screen_shape'], 
#                          a.replay_memory['n_variables'], 
#                          a.replay_memory['n_features'],
#                          type_network ='DQN')
#        
#
#last_states = []
#frame_skip = 4
#screen, v, game_features = e.observe_state(a.variables, a.features)
#eps = a.decrease_eps(1)
#input_screen = a.read_input_state(screen, last_states)
#action = a.act_opt(eps, input_screen)
#r = e.make_action(action, frame_skip)
#screen,v, game_features = e.observe_state(a.variables, a.features)
#input_screen_next = a.read_input_state(screen, last_states, True)
#
#replay_mem.add( screen1=last_states[-1],
#                variables=list(v.values()),
#                features=list(game_features.values()),
#                action=action,
#                reward=r,
#                is_final=e.is_final(),
#                screen2=input_screen_next
#            )






