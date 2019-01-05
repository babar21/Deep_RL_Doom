#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment class
Everything needed to run an experiment (ie a game) :
    - create Vizdoom game
    - define game parameters
    - process game steps
    - display game if needed
    - log game steps
    
"""
# utils
import os
import numpy as np
from logging import getLogger
import time
import math
from collections import namedtuple

# ViZDoom library
from vizdoom import DoomGame, GameVariable
from vizdoom import ScreenResolution, ScreenFormat, Mode

# logger
logger = getLogger()


#%% Experiment class definition

class Experiment(object):
    def __init__(
        self,
        scenario,
        action_builder,
        reward_values=None,
        score_variable='FRAGCOUNT',
        freedoom=True,
        screen_resolution='RES_400X225',
        screen_format='CRCGCB',
        use_screen_buffer=True,
        use_depth_buffer=False,
        labels_mapping='',
        game_features=[],
        mode='PLAYER',
        player_rank=0, players_per_game=1,
        render_hud=False, render_minimal_hud=False,
        render_crosshair=True, render_weapon=True,
        render_decals=False,
        render_particles=False,
        render_effects_sprites=False,
        respawn_protect=True, spawn_farthest=True,
        freelook=False, name='Hubert_Bonnisseur_de_la_Bate', color=0,
        visible=False,
        n_bots=0, use_scripted_marines=None,
        doom_skill=2
    ):
        """
        Create a new game.
        score_variable: indicates in which game variable the user score is
            stored. by default it's in FRAGCOUNT, but the score in ACS against
            built-in AI bots can be stored in USER1, USER2, etc.
        render_decals: marks on the walls
        render_particles: particles like for impacts / traces
        render_effects_sprites: gun puffs / blood splats
        color: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray,
               5 - light brown, 6 - light red, 7 - light blue
        """
        # game resources
        game_filename = '%s.wad' % ('freedoom2' if freedoom else 'Doom2')
        self.scenario_path = os.path.join(RESOURCES_DIR, 'scenarios', '%s.wad' % scenario)
        self.game_path = os.path.join(RESOURCES_DIR, game_filename)

        # check parameters
        assert os.path.isfile(self.scenario_path)
        assert os.path.isfile(self.game_path)
        assert hasattr(GameVariable, score_variable)
        assert hasattr(ScreenResolution, screen_resolution)
        assert hasattr(ScreenFormat, screen_format)
        assert use_screen_buffer or use_depth_buffer
        assert hasattr(Mode, mode)
        assert not (render_minimal_hud and not render_hud)
        assert len(name.strip()) > 0 and color in range(8)
        assert n_bots >= 0
        assert (type(use_scripted_marines) is bool or
                use_scripted_marines is None and n_bots == 0)
        assert 0 <= doom_skill <= 4
        assert 0 < players_per_game
        assert 0 <= player_rank
        
        # screen buffer / depth buffer / labels buffer / mode
        self.screen_resolution = screen_resolution
        self.screen_format = screen_format
        self.use_screen_buffer = use_screen_buffer
        self.use_depth_buffer = use_depth_buffer
        self.labels_mapping = parse_labels_mapping(labels_mapping)
        self.game_features = parse_game_features(game_features, self.logger)
        self.use_labels_buffer = self.labels_mapping is not None
        self.use_game_features = any(self.game_features)
        self.mode = mode
        
        # window visibility
        self.visible = visible

        # actor reward
        ''' used for reward shaping (LSTM & Curiosity A3C) '''
        self.reward_builder = RewardBuilder(self, reward_values)
        
        # number of bots in the game
        self.n_bots = n_bots
        self.use_scripted_marines = use_scripted_marines

        # doom skill (ie difficulty of the game)
        self.doom_skill = doom_skill

        
    def start(self, map_id, episode_time=None, log_events=False):
        """
        Start the game.
        If `episode_time` is given, the game will end after the specified time.
        """
        # Save statistics for this map
        self.statistics[map_id] = {k: 0 for k in self.stat_keys}

        # Episode time
        self.episode_time = episode_time

        # initialize the game
        self.game = DoomGame()
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_doom_game_path(self.game_path)

        # map
        assert map_id > 0
        self.map_id = map_id
        self.game.set_doom_map("map%02i" % map_id)

        # time limit
        if episode_time is not None:
            self.game.set_episode_timeout(int(35 * episode_time))

        # log events that happen during the game (useful for testing)
        self.log_events = log_events

        # game parameters
        args = []

        # screen buffer / depth buffer / labels buffer / mode
        screen_resolution = getattr(ScreenResolution, self.screen_resolution)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_screen_format(getattr(ScreenFormat, self.screen_format))
        self.game.set_depth_buffer_enabled(self.use_depth_buffer)
        self.game.set_labels_buffer_enabled(self.use_labels_buffer or
                                            self.use_game_features)
        self.game.set_mode(getattr(Mode, self.mode))

        # rendering options
        self.game.set_render_hud(self.render_hud)
        self.game.set_render_minimal_hud(self.render_minimal_hud)
        self.game.set_render_crosshair(self.render_crosshair)
        self.game.set_render_weapon(self.render_weapon)
        self.game.set_render_decals(self.render_decals)
        self.game.set_render_particles(self.render_particles)
        self.game.set_render_effects_sprites(self.render_effects_sprites)

        # deathmatch mode
        # players will respawn automatically after they die
        # autoaim is disabled for all players
        args.append('-deathmatch')
        args.append('+sv_forcerespawn 1')
        args.append('+sv_noautoaim 1')

        # respawn invincibility / distance
        # players will be invulnerable for two second after spawning
        # players will be spawned as far as possible from any other players
        args.append('+sv_respawnprotect %i' % self.respawn_protect)
        args.append('+sv_spawnfarthest %i' % self.spawn_farthest)

        # freelook / agent name / agent color
        args.append('+freelook %i' % (1 if self.freelook else 0))
        args.append('+name %s' % self.name)
        args.append('+colorset %i' % self.color)

        # enable the cheat system (so that we can still
        # send commands to the game in self-play mode)
        args.append('+sv_cheats 1')

        # load parameters
        self.args = args
        for arg in args:
            self.game.add_game_args(arg)

        # window visibility
        self.game.set_window_visible(self.visible)

        # available buttons
        self.mapping = add_buttons(self.game, self.action_builder.available_buttons)

        # doom skill (https://zdoom.org/wiki/GameSkill)
        self.game.set_doom_skill(self.doom_skill + 1)

        # start the game
        self.game.init()

        # initialize the game after player spawns
        self.initialize_game()
        
        
        def close(self):
        """
        Close the current experiment.
        """
            self.game.close()
            
        
        


#%% Methods

def parse_game_features(s, logger):
    """
    Parse the game features we want to detect (LSTM) or use as measurements (DPF).
    """
    existing_game_features = ['target', 'enemy', 'health', 'weapon', 'ammo']
    game_feature = []
    for feature in s :
        if feature in existing_game_features:
            game_feature.append(feature)
        else:
            logger.warning("{} is not a feature available!".format(feature))
    return game_feature
