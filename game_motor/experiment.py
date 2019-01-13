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
from collections import namedtuple, defaultdict

# ViZDoom library
import vizdoom as vzd
from vizdoom import DoomGame, GameVariable
from vizdoom import ScreenResolution, ScreenFormat, Mode

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
    'kill_count': GameVariable.KILLCOUNT,
    'item_count' : GameVariable.ITEMCOUNT,
    'death_count' : GameVariable.DEATHCOUNT,
    'damage_count' : GameVariable.DAMAGE_TAKEN,
    'hit_taken' : GameVariable.HITS_TAKEN
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

STAT_KEYS = ['kills', 'deaths', 'suicides', 'frag_count', 'k/d',
             'medikit', 'armor', 'found_weapon', 'ammo', 'health']


#%% Experiment class definition

class Experiment(object):
    """
    Used to perform experiment combined with a Agent 
    Main methods : 
        - 
    """
    
    def __init__(
        self,
        scenario,
        action_builder,
        reward_builder,
        logger,
        living_reward=0,
        custom_reward = False,
        score_variable='FRAGCOUNT',
        game_features=[],
        freedoom=True,
        screen_resolution='RES_400X225',
        screen_format='CRCGCB',
        use_screen_buffer=True,
        use_depth_buffer=False,
        use_labels_buffer=True,
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
        self.reward_builder = reward_builder
        self.living_reward = living_reward
        self.custom_reward = custom_reward
        
        # number of bots in the game
        self.n_bots = n_bots
        self.use_scripted_marines = use_scripted_marines

        # doom skill (ie difficulty of the game)
        self.doom_skill = doom_skill
        
        # bot name
        self.name = name
        
        # action builder
        self.action_builder = action_builder
        
        # save game statistics for each episode (used for model comparison and reward shaping)
        self.stats = {}
        
        # use logging for DEBUG purpose
        self.logger = logger

#==============================================================================
# Game start
#==============================================================================       
  
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
            self.game.set_episode_timeout(episode_time)
        
        # Save statistics for this map
        self.stats[self.map_id ]= []
        
        # log events that happen during the game (useful for testing)
#        self.log_events = log_events

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
#        args.append('-deathmatch')
        args.append('+sv_forcerespawn 1')
        args.append('+sv_noautoaim 1')
        
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
        
        # define basic rewards
        self.game.set_living_reward(self.living_reward)

        # start the game
        self.game.init()

        # initialize the game after player spawns
        self.initialize_game()
        self.logger.info('start_game')


#==============================================================================
# Game statistics
#==============================================================================       
  
    def update_game_properties(self):
        """
        Update game properties.
        """
        # read game variables
        new_v = {k: self.game.get_game_variable(v) for k, v in GAME_FEATURES.items()}
        new_v = {k: (int(v) if v.is_integer() else float(v)) for k, v in new_v.items()}
        
        # update game properties
        self.prev_properties = self.properties
        self.properties = new_v


    def update_game_statistics(self):
        """
        Calculate game statistics and store them in the running stats dict
        """
        stats = self.run_stats
        
        # init r if custom rewards
        r = []
        
        # calculate stats
        # kill
        d = self.properties['kill_count'] - self.prev_properties['kill_count']
        if d > 0:
            r.extend(d*['kill_count'])
            stats['kills'] += d

        # death
        if self.game.is_player_dead():
            r.append('dead')
            stats['deaths'] += 1

        # suicide
        if self.properties['frag_count'] < self.prev_properties['frag_count']:
            r.append('suicide')
            stats['suicides'] += 1

        # found health
        d = self.properties['health'] - self.prev_properties['health']
        if d != 0:
            if d > 0:
                r.append('medikit')
                stats['medikit'] += 1
        stats['health'] = self.properties['health']
        
        # health lost
        d = self.properties['damage_count'] - self.prev_properties['damage_count']
        if d>0:
            r.append('health_lost')
        

        # found armor
        d = self.properties['armor'] - self.prev_properties['armor']
        if d != 0:
            if d > 0:
                r.append('armor')
                stats['armor'] += 1
            
        # found weapon
        if self.prev_properties['sel_weapon'] != self.properties['sel_weapon']:
            r.append('weapon')
            stats['found_weapon'] += 1

        # found / lost ammo
        d = self.properties['sel_ammo'] - self.prev_properties['sel_ammo']
        if self.prev_properties['sel_weapon'] == self.properties['sel_weapon']:
            if d != 0:
                if d > 0:
                    r.append('ammo')
                    stats['ammo'] += 1
                else:
                    r.append('use_ammo')
         
        # auxiliary stats not used for rewards
        stats['frag_count'] = self.properties['frag_count']
        
        
        return r


    def calculate_final_stats(self):
        """
        Calculate the final stats from the running stats
        """
        self.run_stats['k/d'] = self.run_stats['kills'] * 1.0 / max(1, self.run_stats['deaths'])
        
        
        

#==============================================================================
# Game handling
#==============================================================================       
        
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
        self.stats[self.map_id ].append(self.run_stats)
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
#        while self.is_player_dead():
#            logger.warning('Player %i is still dead after respawn.' %
#                           self.params.player_rank)
#            self.respawn_player()

    def respawn_player(self):
        """
        Respawn the player on death.
        """
        assert self.is_player_dead()
        self.game.respawn_player()
#        self.log('Respawn player')
        self.initialize_game()


    def new_episode(self):
        """
        Start a new episode.
        """
        # init new stats for the episode
        self.run_stats = {k: 0 for k in STAT_KEYS}
        # init new game
        self.game.new_episode()
        
        # init episode properties
        self.initialize_game()
        
#        self.log('New episode')
        
    def initialize_game(self):
        """
        Reset game properties
        """
        new_v = {k: self.game.get_game_variable(v) for k, v in GAME_FEATURES.items()}
        new_v = {k: (int(v) if v.is_integer() else float(v)) for k, v in new_v.items()}
        
        self.stats
        self.prev_properties = None
        self.properties = new_v
        
    
    def close(self):
        """
        Close the current experiment.
        """
        self.game.close()

            
    def observe_state(self, variable_names, feature_names):
        """
        Observe the current state of the game.
        """
        # read game state
        screen, variables, game_features = process_game_info(self.game, variable_names, feature_names)
#        last_states.append(GameState(screen, variables, game_features))

        # return the screen and the game features
        return screen, variables, game_features
    
    
    def make_action(self, action, variable_names, feature_names, frame_skip=1, sleep=None):
        """
        Process action and give the next state according to the game motor
        Inputs :
            action :
            frame_skips : nb of frames during which the same action is performed
            sleep : pause game for sleep seconds in order to smooth visualization
        Output :
            reward defined in the game motor or customized
            screen          |
            variables       | of the next state (if not final state)
            game_features   |
        """
        assert frame_skip >= 1

        # convert selected action to the ViZDoom action format
        action = self.action_builder.get_action(action)
        
        # smooth visualization if needed for make
        if self.visible:
            r=0
            for _ in range(frame_skip):
                r += self.game.make_action(action)
                # death or episode finished
                if self.is_player_dead() or self.is_episode_finished():
                    break
                # sleep for smooth visualization
                if sleep is not None:
                    time.sleep(sleep)
        else:
            r = self.game.make_action(action, frame_skip)
        
        # observe resulting state
        if not self.is_final():
            screen, variables, game_features = self.observe_state(variable_names, feature_names)
        else :
            screen = None
            variables = None
            game_features = None
        # update game statistics and return custom rewards
        self.update_game_properties()
        list_r = self.update_game_statistics()
        if self.custom_reward and self.reward_builder :
            r = self.reward_builder.get_reward(list_r)
        
        return r, screen, variables, game_features
        


#%% Methods

def process_game_info(game, variables_names, features_name):
    """
    Get state from the vizdoom game object
    """
    state = game.get_state()
#    n = state.number
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


def process_game_statistics_raw(list_dict):
    """
    Flatten list of dict to get learning curve for all the parameters
    """
    result = defaultdict(list)
    for dic in list_dict:
        for key in STAT_KEYS:
            result[key].append(dic[key])
    return result
        

def process_game_statistics(dict_of_stats):   
    """
    Process stats for each maps processed
    """
    result = dict()
    for map_id, list_dict in dict_of_stats.items():
        result[map_id] = process_game_statistics_raw(list_dict)
    return result
    
    
    
    
    
    
    