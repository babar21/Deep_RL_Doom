#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parsers :
    - different kind of parsers for the neural networks parameters
"""

def parse_image_params(image_params):
    """
    parse dict to build the image stream in the DFP nn
    """
    if 'screen_input_size' not in image_params:
        raise Exception('You must provide the screen input size for the neural network')
    
    if ('s1' not in image_params) or ('s2' not in image_params) or ('s3' not in image_params) or ('s4' not in image_params):
        s1 = dict()
        s2 = dict()
        s3 = dict()
        s4 = dict()
        if image_params['screen_input_size'] == (84,84,1):
            s1['channel']=32
            s1['kernel']=8
            s1['stride']=4
            s2['channel']=64
            s2['kernel']=4
            s2['stride']=2
            s3['channel']=64
            s3['kernel']=3
            s3['stride']=1
            s4['output']=512
        elif image_params['screen_input_size'] == (128,128,1):
            s1['channel']=32
            s1['kernel']=8
            s1['stride']=4
            s2['channel']=64
            s2['kernel']=4
            s2['stride']=2
            s3['channel']=128
            s3['kernel']=3
            s3['stride']=1
            s4['output']=1024
    else :
        s1 = image_params['s1']
        s2 = image_params['s2']
        s3 = image_params['s3']
        s4 = image_params['s4']
    
    return image_params['screen_input_size'], s1, s2, s3, s4


def parse_measure_params(measure_params):
    """
    parse dict to build the measurement stream in the DFP nn
    """
    if 'measure_input_size' not in measure_params:
        raise Exception('You must provide the measure input size for the neural network')
    
    if ('m1' not in measure_params) or ('m2' not in measure_params) or ('m3' not in measure_params):
        m1 = dict()
        m2 = dict()
        m3 = dict()
        m1['output']=128
        m2['output']=128
        m3['output']=128
        
    else :
        m1 = measure_params['m1']
        m2 = measure_params['m2']
        m3 = measure_params['m3']
    
    return measure_params['measure_input_size'], m1, m2, m3
    
    
def parse_goal_params(goal_params):
    """
    parse dict to build the goal stream in the DFP nn
    """
    if 'goal_input_size' not in goal_params:
        raise Exception('You must provide the goal input size for the neural network')
    
    if ('g1' not in goal_params) or ('g2' not in goal_params) or ('g3' not in goal_params):
        g1 = dict()
        g2 = dict()
        g3 = dict()
        g1['output']=128
        g2['output']=128
        g3['output']=128
        
    else :
        g1 = goal_params['g1']
        g2 = goal_params['g2']
        g3 = goal_params['g3']
    
    return goal_params['goal_input_size'], g1, g2, g3


def parse_action_params(action_params):
    """
    parse dict to build the action stream in the DFP nn
    """
    if 'nb_actions' not in action_params:
        raise Exception('You must provide the number of actions for the neural network')
    if 'a1' not in action_params :
        a1 = dict()
        a1['output']=512
    else :
        a1 = action_params['a1']
        
    return action_params['nb_actions'], a1


def parse_expectation_params(expectation_params):
    """
    parse dict to build the expectation stream in the DFP nn
    """
    if 'e1' not in expectation_params :
        e1 = dict()
        e1['output']=512
    else :
        e1 = expectation_params['e1']
        
    return e1


def parse_image_params_dqn(image_params):
    s1 = image_params['s1']
    s2 = image_params['s2']
    s3 = image_params['s3']
    
    return image_params['screen_input_size'], s1, s2, s3
    
    
    
    
    
    
  