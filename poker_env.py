import gymnasium as gym
from enum import Enum
from player import RLPlayer, PlayerCycle

import numpy as np


# fix this
class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE_5 = 2
    RAISE_10 = 3
    RAISE_20 = 4
    RAISE_50 = 5


INITIAL_MONEY = 100
SMALL_BLIND = 1


class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            'visible_cards_win_probability': gym.spaces.Box(low=0, high=1, shape=(1,), dtype='float32'),
            'players_playing': gym.spaces.MultiBinary(6),
            'money': gym.spaces.Box(low=0, high=INITIAL_MONEY, dtype='int32'),
            'round_bets': gym.spaces.Box(low=0, high=INITIAL_MONEY, shape=(4, 6), dtype='int32'),
            'player_ind': gym.spaces.Discrete(6),
        })

    def reset(self, player_cycle: PlayerCycle):
        self.player_cycle = player_cycle
        self.n_players = len(player_cycle.get_player_list())
        self.players_playing = np.array([True] * 6)
        self.money = np.array([INITIAL_MONEY] * 6)
        self.round_bets = np.zeros((4, 6), dtype='int32')
        self.round_bets[0, 0] = SMALL_BLIND
        self.round_bets[0, 1] = 2 * SMALL_BLIND
        self.player_ind = np.zeros(6, dtype='int32')
        self.player_ind[0] = 1

        # playing until all non-RL players have played
        while type(player_cycle.get_player()) != RLPlayer:
            self.apply_action(player_cycle.get_player_index(), player_cycle.get_player().get_action(self))
            player_cycle.next_player()

        return {
            'players_playing': self.players_playing,
            'money': self.money,
            'round_bets': self.round_bets,
        }

    def step(self, action):
        for _ in range(len(self.player_cycle.get_player_list())):
            if not self.players_playing[self.player_cycle.get_player_index()]:
                self.player_cycle.next_player()
                continue
            self.apply_action(self.player_cycle.get_player_index(), action)
            self.player_cycle.next_player()

    def apply_action(self, player, action):
        if action == Action.FOLD:
            self.players_playing[player] = False
        elif action == Action.CALL:
            pass
        elif action == Action.RAISE_5:
            pass
        elif action == Action.RAISE_10:
            pass
        elif action == Action.RAISE_20:
            pass
        elif action == Action.RAISE_50:
            pass
        else:
            raise ValueError("Invalid action")
