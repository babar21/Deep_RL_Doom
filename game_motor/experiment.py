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
from logging import getLogger
import time
from collections import namedtuple

# ViZDoom library
import vizdoom as vzd
from vizdoom import DoomGame, GameVariable
from vizdoom import ScreenResolution, ScreenFormat, Mode

# logger
logger = getLogger()

# Custom type
GameState = namedtuple('State', ['screen', 'variables', 'features'])

# Global variables 

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),'experiments')

GAME_FEATURES = {
    'frag_count' : GameVariable.FRAGCOUNT,
    'health' : GameVariable.HEALTH,
    'armor' : GameVariable.ARMOR,
    'sel_weapon': GameVariable.SELECTED_WEAPON,
    'sel_ammo': GameVariable.SELECTED_WEAPON_AMMO,
    'ammo': GameVariable.AMMO2,
    'shells': GameVariable.AMMO3,
}

GAME_VARIABLES = {
'ENNEMY' : set([
    'MarineBFG', 'MarineBerserk', 'MarineChaingun', 'MarineChainsaw',
    'MarineFist', 'MarinePistol', 'MarinePlasma', 'MarineRailgun',
    'MarineRocket', 'MarineSSG', 'MarineShotgun',
    'Demon'
]),
'HEALTH' : set([
    'ArmorBonus', 'BlueArmor', 'GreenArmor', 'HealthBonus',
    'Medikit', 'Stimpack'
]),
'WEAPON' : set([
    'Pistol', 'Chaingun', 'RocketLauncher', 'Shotgun', 'SuperShotgun',
    'PlasmaRifle', 'BFG9000', 'Chainsaw'
]),
'AMMO' : set([
    'Cell', 'CellPack', 'Clip', 'ClipBox', 'RocketAmmo', 'RocketBox',
    'Shell', 'ShellBox'
])

    }
    
EMBED_GAME_VARIABLES = {
        'ENNEMY' : 0,
        'HEALTH' : 1,
        'WEAPON' : 2,        
        'AMMO' : 3
        }

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
        use_labels_buffer=True,
        game_features=[],
        mode='PLAYER',
        player_rank=0, players_per_game=1,
        render_hud=False, render_minimal_hud=False,
        render_crosshair=True, render_weapon=True,
        render_decals=False,
        render_particles=False,
        render_effects_sprites=False,
        respawn_protect=True, spawn_farthest=True,
        name='Hubert_Bonnisseur_de_la_Bate',
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
        game_filename = 'freedoom2.wad'
        self.scenario_path = os.path.join(PATH, 'scenarios/{}.wad'.format(scenario))
        self.game_path = os.path.join(PATH, game_filename)

        # check parameters
        assert os.path.isfile(self.scenario_path)
        assert os.path.isfile(self.game_path)
        assert hasattr(GameVariable, score_variable)
        assert hasattr(ScreenResolution, screen_resolution)
        assert hasattr(ScreenFormat, screen_format)
        assert use_screen_buffer or use_depth_buffer
        assert hasattr(Mode, mode)
        assert not (render_minimal_hud and not render_hud)
        assert len(name.strip()) > 0 
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
        self.game_features = parse_game_features(game_features, logger)
        self.use_labels_buffer = use_labels_buffer
        self.use_game_features = any(self.game_features)
        self.mode = mode
        
        # rendering options
        self.render_hud = render_hud
        self.render_minimal_hud = render_minimal_hud
        self.render_crosshair = render_crosshair
        self.render_weapon = render_weapon
        self.render_decals = render_decals
        self.render_particles = render_particles
        self.render_effects_sprites = render_effects_sprites
        
        # window visibility
        self.visible = visible

        # actor reward
        ''' used for reward shaping (LSTM & Curiosity A3C) '''
#        self.reward_builder = RewardBuilder(self, reward_values)
        
        # number of bots in the game
        self.n_bots = n_bots
        self.use_scripted_marines = use_scripted_marines

        # doom skill (ie difficulty of the game)
        self.doom_skill = doom_skill
        
        # bot name
        self.name = name
        
        # action builder
        self.action_builder = action_builder

        
    def start(self, map_id, episode_time=None, log_events=False):
        """
        Start the game.
        If `episode_time` is given, the game will end after the specified time.
        """

        # Episode time
        self.episode_time = episode_time

        # initialize the game
        self.game = DoomGame()
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_doom_game_path(self.game_path)

        # map
        assert map_id > 0
        self.map_id = map_id
        self.game.set_doom_map('map{:02d}'.format(map_id))

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
        self.game.set_labels_buffer_enabled(self.use_labels_buffer)
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
#        args.append('+sv_respawnprotect %i' % self.respawn_protect)
#        args.append('+sv_spawnfarthest %i' % self.spawn_farthest)

        # agent name 
        args.append('+name %s' % self.name)


        # load parameters
        self.args = args
        for arg in args:
            self.game.add_game_args(arg)

        # window visibility
        self.game.set_window_visible(self.visible)

        # define available buttons
        self.action_builder.set_buttons(self.game)

        # doom skill (https://zdoom.org/wiki/GameSkill)
        self.game.set_doom_skill(self.doom_skill + 1)

        # start the game
        self.game.init()

        # initialize the game after player spawns
#        self.initialize_game()
        
        
    def is_player_dead(self):
        """
        Detect whether the player is dead.
        """
        return self.game.is_player_dead()

    def is_episode_finished(self):
        """
        Return whether the episode is finished.
        This should only be the case after the episode timeout.
        """
        return self.game.is_episode_finished()

    def is_final(self):
        """
        Return whether the game is in a final state.
        """
        return self.is_player_dead() or self.is_episode_finished()
    
    def reset(self):
        """
        Reset the game if necessary. This can be because:
            - we reach the end of an episode (we restart the game)
            - because the agent is dead (we make it respawn)
        """
        self.count_non_forward_actions = 0
        # if the player is dead
        if self.is_player_dead():
            # respawn it (deathmatch mode)
            if self.episode_time is None:
                self.respawn_player()
            # or reset the episode (episode ends when the agent dies)
            else:
                self.new_episode()

        # start a new episode if it is finished
        if self.is_episode_finished():
            self.new_episode()

        # deal with a ViZDoom issue
        while self.is_player_dead():
            logger.warning('Player %i is still dead after respawn.' %
                           self.params.player_rank)
            self.respawn_player()

    def new_episode(self):
        """
        Start a new episode.
        """
        assert self.is_episode_finished() or self.is_player_dead()
        self.game.new_episode()
        self.log('New episode')
#        self.initialize_game()
    
    def close(self):
        """
        Close the current experiment.
        """
        self.game.close()
 
            
    def observe_state(self, variables_names, features_name):
        """
        Observe the current state of the game.
        """
        # read game state
        screen, variables, game_features = process_game_info(self.game, variables_names, features_name)
#        last_states.append(GameState(screen, variables, game_features))

        # return the screen and the game features
        return screen, variables, game_features
    
    
    def make_action(self, action, frame_skip=1, sleep=None):
        """
        Process action and give the next state according to the game motor
        Inputs :
            action :
            frame_skips : nb of frames during which the same action is performed
            sleep : pause game for sleep seconds in order to smooth visualization
        """
        assert frame_skip >= 1

        # convert selected action to the ViZDoom action format
        action = self.action_builder.get_action(action)
        
        # smooth visualization if needed
        if self.visible:
            for _ in range(frame_skip):
                self.game.make_action(action)
                # death or episode finished
                if self.is_player_dead() or self.is_episode_finished():
                    break
                # sleep for smooth visualization
                if sleep is not None:
                    time.sleep(sleep)
        else:
            self.game.make_action(action, frame_skip)
        
            
        


#%% Methods

def process_game_info(game, variables_names, features_name):
    """
    Get state from the vizdoom game object
    """
    state = game.get_state()
    n = state.number
    screen = state.screen_buffer # retrieve the game screen in full definition and colors
    variables = get_variables(state, variables_names)
    game_features = get_features(game, features_name)
    
    return screen, variables, game_features


def get_variables(state, variables_name):
    """
    Get selected game variables from the game object
    Returns a dict(variable : value)
    """
    variable_dict = dict.fromkeys(variables_name,0.)
    ava_variables = parse_game_variables(state.labels)
    for variable in variables_name :
        if EMBED_GAME_VARIABLES [variable] in ava_variables :
            variable_dict[variable] = 1.
    return variable_dict


def get_features(game, features_name):
    """
    Get selected game features from the game object
    Returns a dict(features : value)
    """
    feature_dict = dict.fromkeys(features_name,0.)
    for features in features_name :
        feature_dict[features] = game.get_game_variable(GAME_FEATURES[features])
    return feature_dict
    
   
def parse_game_variables(list_labels):
    """
    Embedded each variable with an integer
    """
    l = []
    for obj in list_labels:
        if obj.object_name in GAME_VARIABLES['ENNEMY']:
            l.append(0)
        elif obj.object_name in GAME_VARIABLES['HEALTH']:
            l.append(1)
        elif obj.object_name in GAME_VARIABLES['WEAPON']:
            l.append(2)
        elif obj.object_name in GAME_VARIABLES['AMMO']:
            l.append(3)
    return set(l)
    
    
def parse_game_features(s, logger):
    """
    Parse the game features we want to detect (LSTM) or use as measurements (DPF).
    """
    existing_game_features = list(GAME_FEATURES.keys())
    game_feature = []
    for feature in s :
        if feature in existing_game_features:
            game_feature.append(feature)
        else:
            logger.warning("{} is not a feature available!".format(feature))
    return game_feature
