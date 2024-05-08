from enum import Enum
from random import shuffle

import gymnasium as gym
import numpy as np

from player import RLPlayer, PlayerCycle, Player
from poker_rules import Deck

from poker_lib import get_chances


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
ITERATIONS = 1000

class PokerEnv(gym.Env):
    def __init__(self, player_list: list[Player]):
        super(PokerEnv, self).__init__()
        self.player_list = player_list
        self.deck = Deck()
        self.is_done = False
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            'visible_cards_win_probability': gym.spaces.Box(low=0, high=1, shape=(1,), dtype='float32'),
            'money': gym.spaces.Box(low=0, high=INITIAL_MONEY, dtype='int32'),
            'players_playing': gym.spaces.MultiBinary(6),
            'round_bets': gym.spaces.Box(low=0, high=INITIAL_MONEY, shape=(4, 6), dtype='int32'),
            'player_ind': gym.spaces.MultiBinary(6),
        })
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # change the order of the players
        shuffle(self.player_list)
        self.player_cycle = PlayerCycle(self.player_list)

        self.players_playing = np.array([True] * 6)
        self.money = np.array([INITIAL_MONEY] * 6)
        self.round_bets = np.zeros((4, 6), dtype='int32')
        self.round_index = 0

        # small blind and big blind
        self.round_bets[0, 0] = SMALL_BLIND
        self.round_bets[0, 1] = 2 * SMALL_BLIND
        self.money[0] -= SMALL_BLIND
        self.player_cycle.next_player()
        self.money[1] -= 2 * SMALL_BLIND
        self.player_cycle.next_player()

        self.deck.shuffle()

        self.user_hands = [self.deck.draw_n(2) for _ in range(6)]
        self.user_hands_encoded = [str(hand) for hand in self.user_hands]
        self.winning_probabilities = [0] * 6
        self.table_cards = []

        self._calculate_winning_probabilities()

        self.round_index = 0

        # playing until all non-RL players have played
        while type(self.player_cycle.get_player()) != RLPlayer:
            self.apply_action(self.player_cycle.get_player_index(), self.player_cycle.get_player().get_action(self))
            self.player_cycle.next_player()

        self.rl_player_id = self.player_cycle.get_player_index()
        # one hot encoding of the player index
        self.encoded = np.zeros(6)
        self.encoded[self.rl_player_id] = 1
        observation = {
            'visible_cards_win_probability': self.winning_probabilities[self.rl_player_id],
            'money': self.money[self.rl_player_id],
            'players_playing': self.players_playing,
            'round_bets': self.round_bets,
            'player_ind': self.encoded,
        }
        return observation, {}

    def step(self, action: Action):
        # first action is played by the RL player
        self.apply_action(self.player_cycle.get_player_index(), action)
        self.player_cycle.next_player()

        # play other players
        for _ in range(len(self.player_list) - 1):
            if not self.players_playing[self.player_cycle.get_player_index()]:
                self.player_cycle.next_player()
                continue
            self.apply_action(self.player_cycle.get_player_index(),
                              self.player_list[self.player_cycle.get_player_index()].get_action(self))
            self.player_cycle.next_player()

        observation = {
            'visible_cards_win_probability': self.winning_probabilities[self.rl_player_id],
            'money': self.money[self.rl_player_id],
            'players_playing': self.players_playing,
            'round_bets': self.round_bets,
            'player_ind': self.encoded,
        }
        reward = 0
        if self.is_done:
            reward = self.money[self.rl_player_id] if self.money[self.rl_player_id] > 0 else -20
        return observation, reward, self.is_done, False, {}


    def apply_action(self, player: int, action: Action):
            if action == Action.FOLD:
                self.players_playing[player] = False
                # if the player is an RL player, the game doesn't need to be simulated anymore
                if isinstance(self.player_list[player], RLPlayer):
                    self.is_done = True
                if sum(self.players_playing) == 1:
                    self._resolve_game()
            elif action == Action.CALL:
                self.money[player] -= max(self.round_bets[self.round_index]) - self.round_bets[self.round_index, player]
                self.round_bets[self.round_index, player] = max(self.round_bets[self.round_index])

                # check if all playing players have the same bet
                playing_bets = [self.round_bets[self.round_index, i] for i in range(6) if self.players_playing[i]]
                if playing_bets.count(playing_bets[0]) == len(playing_bets):
                    self.next_round()
            elif action == Action.RAISE_5:
                self._raise(player, 5)
            elif action == Action.RAISE_10:
                self._raise(player, 10)
            elif action == Action.RAISE_20:
                self._raise(player, 20)
            elif action == Action.RAISE_50:
                self._raise(player, 50)
            else:
                raise ValueError("Invalid action")
    def next_round(self):
        self.round_index += 1
        if self.round_index == 1:
            self.table_cards = self.deck.draw_n(3)
        elif self.round_index == 2:
            self.table_cards.append(self.deck.draw_n(1))
        elif self.round_index == 3:
            self.table_cards.append(self.deck.draw_n(1))
        if self.round_index == 4:
            self._resolve_game()
        self._calculate_winning_probabilities()

    def _calculate_winning_probabilities(self):
        encoded_table_cards = [str(card) for card in self.table_cards]
        playing = self.players_playing.sum()
        self.winning_probabilities = [get_chances(player_cards + encoded_table_cards, playing, ITERATIONS) for player_cards in self.user_hands_encoded]

    def _resolve_game(self):
        winning_player = 0  # TODO calculate winning player
        self.money[winning_player] += sum(self.round_bets[3])

    def _raise(self, player: int, amount: int):
        # if player can't raise the full amount, raise as much as possible
        amount = min(amount, self.money[player])
        self.round_bets[self.round_index, player] += amount
        self.money[player] -= amount
