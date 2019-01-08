#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent class
Everything needed for an agent to :
    - get input from the environment
    - choose an action according to a policy
    
"""

from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract class, wrapping all the agents
    """

    @abstractmethod
    def act_opt(self,*args):
        """
        Uses input games parsed by read_input_state and returns an action (which 
        will be parsed by the make_action method)
        """
        pass
    
    
    @abstractmethod
    def read_input_state(self, screen, game_features):
        """
        Takes inputs as given by the experiment class and returns inputs working fine
        for the given bot
        """
        pass


    @abstractmethod
    def train(self):
        """
        Takes an Experiment class method and train the model according to the agent
        training framework
        """
        pass