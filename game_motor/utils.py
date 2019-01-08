#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions (convert, plot...)

"""

def get_subkey(dict_of_dict):
    """ return all the key of a dict of dict in a list """
    new_key = []
    for key, value in dict_of_dict.items():
        new_key.extend(value)
    return list(set(new_key))


def flatten_dict(dict_of_dict):
    new_dict = {}
    for key, value in dict_of_dict.items():
        for name in value:
            new_dict[name] = key
    return new_dict


def dict_to_list_of_list(dict_of_dict):
    l = []
    for key, value in dict_of_dict.items():
        l.append(value)
    return l