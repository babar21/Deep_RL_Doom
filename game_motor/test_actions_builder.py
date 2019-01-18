#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:48:40 2019

@author: bastien
"""

from utils import get_subkey, flatten_dict
import itertools
import actions_builder

#%% Global variables

ALL_AVAILABLE_ACTIONS = {
    'turn_lr': ['TURN_LEFT', 'TURN_RIGHT'],
    'move_fb': ['MOVE_FORWARD', 'MOVE_BACKWARD'],
    'move_lr': ['MOVE_LEFT', 'MOVE_RIGHT'],
    'shoot': ['ATTACK'],
    'run' : ['SPEED']
}

list_action =  get_subkey(ALL_AVAILABLE_ACTIONS)
list_action.append('test') # insert non existing action

def parse_available_button(list_action):  
    """
    Assert actions are possible and create all possible combinations of available actions
    """
    all_actions = get_subkey(ALL_AVAILABLE_ACTIONS)
    reversed_dict = flatten_dict(ALL_AVAILABLE_ACTIONS)
    a_action = []
    new_dict_act = defaultdict(list)
    for action in list_action :
        if action in all_actions:
            a_action.append(action)
            new_dict_act[reversed_dict[action]].append(action)
#        else :
#            logger.warning("{} is not an action available!".format(action))
    # check if at least one action was correct
    assert a_action
    
    return a_action, new_dict_act
        

def create_available_actions():
    """
    Create all possible actions for the bot
    """
    available_act = []
    l = dict_to_list_of_list(AVAILABLE_ACTIONS)
    possible_subset = list(itertools.product(*l))
    for sub in possible_subset:
        for i in range(1,len(sub)+1):
            available_act.extend(list(itertools.combinations(sub, i)))
    available_act_g = list(set([tuple(sorted(tp)) for tp in available_act]))
    return available_act_g

#available_buttons, dico = parse_available_button(list_action)

act = Action(list_action)
