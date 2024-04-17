import gymnasium as gym
from enum import Enum

import numpy as np


# fix this
class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    CHECK = 3
    BET = 4
    ALLIN = 5

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

    def reset(self):

        self.players_playing = np.array([True] * 6)
        self.money = np.array([INITIAL_MONEY] * 6)
        self.round_bets = np.zeros((4, 6), dtype='int32')
        self.round_bets[0, 0] = SMALL_BLIND
        self.round_bets[0, 1] = 2 * SMALL_BLIND
        self.player_ind = np.zeros(6, dtype='int32')
        self.player_ind[0] = 1


        return {
            'players_playing': self.players_playing,
            'money': self.money,
            'round_bets': self.round_bets,
        }


    def step(self, action):
        pass


