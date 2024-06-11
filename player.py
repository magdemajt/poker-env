import math
import random
from abc import ABC, abstractmethod
from typing import Any
from poker_rules import Action

import numpy as np


class Player(ABC):
    @abstractmethod
    def get_action(self, observation: dict[str, Any], action_space: Any):
        raise NotImplementedError

    @abstractmethod
    def update(self, reward):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class RandomPlayer(Player):
    def get_action(self, observation: dict[str, Any], action_space: Any):
        return action_space.sample()

    def update(self, reward):
        pass

    def reset(self):
        pass


class RLPlayer(Player):
    def get_action(self, observation: dict[str, Any], action_space: Any):
        raise NotImplementedError

    def update(self, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class HeuristicPlayer(Player):

    def get_action(self, observation: dict[str, Any], action_space):
        win_prob = observation['visible_cards_win_probability'][0]
        money = observation['money'][0]
        total_on_the_table = np.sum(observation['round_bets'])

        randomized = random.randint(1, 100)

        if randomized == 69:
            return Action.RAISE_20


        if money < 50:
            # TODO if the player starts this should be adjusted
            if win_prob >= 0.5:
                return Action.RAISE_20
            else:
                return Action.FOLD

        def get_random_raise_call(action: Action):
            return random.choice([action, Action.CALL])

        if math.ceil(total_on_the_table * win_prob) > 20:
            return get_random_raise_call(Action.RAISE_20)
        elif math.ceil(total_on_the_table * win_prob) > 10:
            return get_random_raise_call(Action.RAISE_10)
        elif math.ceil(total_on_the_table * win_prob) > 50:
            return get_random_raise_call(Action.RAISE_5)
        elif math.ceil(total_on_the_table * win_prob) > 1:
            return get_random_raise_call(Action.RAISE_1)
        else:
            return get_random_raise_call(Action.FOLD)


    def update(self, reward):
        pass

    def reset(self):
        pass


class PlayerCycle:
    def __init__(self, players_list):
        self.players_list = players_list
        self.current_player = 0

    def next_player(self):
        self.current_player += 1
        self.current_player %= len(self.players_list)
        return self.current_player

    def get_player(self):
        return self.players_list[self.current_player]

    def get_player_index(self):
        return self.current_player

    def get_player_list(self):
        return self.players_list

    def reset(self):
        self.current_player = 0
