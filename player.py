from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, money):
        self.money = money

    @abstractmethod
    def get_action(self, env):
        raise NotImplementedError

    @abstractmethod
    def update(self, reward):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class RandomPlayer(Player):
    def get_action(self, env):
        return env.action_space.sample()

    def update(self, reward):
        pass

    def reset(self):
        pass

class RLPlayer(Player):
    def get_action(self, env):
        pass

    def update(self, reward):
        pass

    def reset(self):
        pass


class HeuristicPlayer(Player):

    def get_action(self, env):
        pass

    def update(self, reward):
        pass

    def reset(self):
        pass


class PlayerCycle():
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