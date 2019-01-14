#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward class : everything needed to design custom reward (used for reward shaping for instance)
"""

DEFAULT_REWARD_VALUES = {
    'BASE_REWARD': 0.,
    'DISTANCE': 0.,
    'kill_count': 5.,
    'dead': -5.,
    'suicide': -5.,
    'medikit': 1.,
    'armor': 1.,
    'health_lost': -1.,
    'weapon': 1.,
    'ammo': 1.,
    'use_ammo': -0.2,
}


class Reward(object):
    """
    Define and calculate custom rewards
    """
    def __init__(self, dic_reward=None, default=True):
        if dic_reward : 
            self.default_reward = parse_reward_name(dic_reward, default)
        else :
            self.default_reward = DEFAULT_REWARD_VALUES
        
    def get_reward(self, list_r):
        """
        Calculate reward from the list of changed game statistics
        """
#        assert all(set(list_r)) in list(self.default_reward)
        r=0
        for elt in list_r:
            r += self.default_reward.get(elt,0)
        return r
        
     
def parse_reward_name(dic, default):
    """
    parse available rewards name
    """
    dic_reward = dict()
    for name, value in dic.items():
        if name in DEFAULT_REWARD_VALUES:
            if default : 
                dic_reward[name] = DEFAULT_REWARD_VALUES[name]
            else :
                dic_reward[name] = value
    return dic_reward       
            
            
            
            
            