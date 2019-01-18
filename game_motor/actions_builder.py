#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action setup : 
    - to describe available action
    - to translate integer action to vizdoom action 
"""

from utils import get_subkey, dict_to_list_of_list, flatten_dict
from logging import getLogger
import itertools
from collections import defaultdict
from vizdoom import Button 

# logger
logger = getLogger()

#%% Global variables

ALL_AVAILABLE_ACTIONS = {
    'turn_lr': ['TURN_LEFT', 'TURN_RIGHT'],
    'move_fb': ['MOVE_FORWARD', 'MOVE_BACKWARD'],
    'move_lr': ['MOVE_LEFT', 'MOVE_RIGHT'],
    'shoot': ['ATTACK'],
    'run' : ['SPEED']
}


VIZDOOM_ACTIONS = {
        'TURN_LEFT' : Button.TURN_LEFT,
        'TURN_RIGHT' : Button.TURN_RIGHT,
        'MOVE_FORWARD' : Button.MOVE_FORWARD,
        'MOVE_BACKWARD' : Button.MOVE_BACKWARD,
        'MOVE_LEFT' : Button.MOVE_LEFT,
        'MOVE_RIGHT' : Button.MOVE_RIGHT,
        'ATTACK' : Button.ATTACK,
        'SPEED' : Button.SPEED
        }


#%% Class
class Action(object):
    """
    """
    def __init__(self, list_action):
        self.available_buttons, dict_action = parse_available_button(list_action)
        self.available_actions = self.create_available_actions(dict_action)
        self.n_actions = len(self.available_actions)
        self.doom_action = self.create_action_doom()
        
    
    def create_available_actions(self, dict_action):
        """
        Create all possible actions for the bot
        """
        available_act = []
        l = dict_to_list_of_list(dict_action)
        possible_subset = list(itertools.product(*l))
        for sub in possible_subset:
            for i in range(1,len(sub)+1):
                available_act.extend(list(itertools.combinations(sub, i)))
        available_act = list(set([tuple(sorted(tp)) for tp in available_act]))
        return available_act
        
        
    def create_action_doom(self):
        """
        Return vizdoom interpretable action from string action list
        """
        action_doom = []
        for action in self.available_actions :
            doom_action = [button in action for button in self.available_buttons]
            action_doom.append(doom_action)
        return action_doom
    
    
    def get_action(self, action_int):
        """
        Return vizdoom interpretable action from an action embedding
        """
        return self.doom_action[action_int]

    
    def set_buttons(self, game):
        """
        Set available buttons when init the game
        """
        for button in self.available_buttons:
            game.add_available_button(VIZDOOM_ACTIONS[button])
            
        
#%% Methods

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
        else :
            logger.warning("{} is not an action available!".format(action))
    # check if at least one action was correct
    assert a_action
    
    return a_action, new_dict_act
        

            
        
        
        
   