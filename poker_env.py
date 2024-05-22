from random import shuffle

import gymnasium as gym
import numpy as np

from player import RLPlayer, PlayerCycle
from poker_lib import get_chances, get_win_indices
from poker_rules import Deck, Action
from ray.rllib.env import EnvContext

INITIAL_MONEY = 100
SMALL_BLIND = 1
ITERATIONS = 1000


class PokerEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super(PokerEnv, self).__init__()
        self.player_list = config["player_list"]
        self.deck = Deck()
        self.is_done = False
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            'visible_cards_win_probability': gym.spaces.Box(low=0, high=1, shape=(1,), dtype='float32'),
            'money': gym.spaces.Box(low=0, high=np.inf, dtype='int32'),
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
        self.is_done = False

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
        self.user_hands_encoded = [[str(card) for card in hand] for hand in self.user_hands]
        self.winning_probabilities = [0] * 6
        self.table_cards = []

        self._calculate_winning_probabilities()

        self.round_index = 0

        # playing until all non-RL players have played
        while type(self.player_cycle.get_player()) != RLPlayer:
            self._play_non_rl_player(self.player_cycle.get_player_index())

        self.rl_player_id = self.player_cycle.get_player_index()
        # one hot encoding of the player index
        self.encoded_rl_id = np.zeros(6)
        self.encoded_rl_id[self.rl_player_id] = 1
        observation_rl = self._player_observation(self.rl_player_id, True)
        return observation_rl, {}

    def step(self, action: Action):
        # first action is played by the RL player
        self.apply_action(self.player_cycle.get_player_index(), action)
        self.player_cycle.next_player()

        # play other players
        for _ in range(len(self.player_list) - 1):
            self._play_non_rl_player(self.player_cycle.get_player_index())

        observation_rl = self._player_observation(self.rl_player_id, True)
        reward = 0
        if self.is_done:
            reward = self.money[self.rl_player_id] if self.money[self.rl_player_id] > 0 else -20
        return observation_rl, reward, self.is_done, False, {}

    def apply_action(self, player: int, action: Action):
        if self.money[player] == 0:
            action = Action.CALL.value
        if action == Action.FOLD.value:
            self.players_playing[player] = False
            # if the player is an RL player, the game doesn't need to be simulated anymore
            if isinstance(self.player_list[player], RLPlayer):
                self.is_done = True
            if sum(self.players_playing) == 1:
                self._resolve_game()
        elif action == Action.CALL.value:
            self.money[player] -= max(self.round_bets[self.round_index]) - self.round_bets[self.round_index, player]
            self.round_bets[self.round_index, player] = max(self.round_bets[self.round_index])

            # check if all playing players have the same bet
            playing_bets = [self.round_bets[self.round_index, i] for i in range(6) if self.players_playing[i]]
            if playing_bets.count(playing_bets[0]) == len(playing_bets):
                self.next_round()
        elif action == Action.RAISE_5.value:
            self._raise(player, 5)
        elif action == Action.RAISE_10.value:
            self._raise(player, 10)
        elif action == Action.RAISE_20.value:
            self._raise(player, 20)
        elif action == Action.RAISE_50.value:
            self._raise(player, 50)
        else:
            raise ValueError(f"Invalid action, action: {action}")

    def next_round(self):
        self.round_index += 1
        if self.round_index == 1:
            self.table_cards = self.deck.draw_n(3)
        elif self.round_index == 2:
            self.table_cards += self.deck.draw_n(1)
        elif self.round_index == 3:
            self.table_cards += self.deck.draw_n(1)
        elif self.round_index == 4:
            self._resolve_game()
        self._calculate_winning_probabilities()

    def _calculate_winning_probabilities(self):
        encoded_table_cards = [str(card) for card in self.table_cards]
        playing = self.players_playing.sum()
        self.winning_probabilities = [get_chances(player_cards + encoded_table_cards, playing, ITERATIONS) for
                                      player_cards in self.user_hands_encoded]

    def _resolve_game(self):

        still_playing = [player for player in self.players_playing]
        winning_players = get_win_indices([str(card) for card in self.table_cards], self.user_hands_encoded,
                                          still_playing)
        prize_per_winner = np.sum(self.round_bets) // len(winning_players)
        for winner in winning_players:
            self.money[winner] += prize_per_winner
        self.is_done = True

    def _raise(self, player: int, amount: int):
        # if player can't raise the full amount, raise as much as possible
        max_round_bet = max(self.round_bets[self.round_index])
        new_max = min(max_round_bet + amount, self.money[player])
        bet = new_max - self.round_bets[self.round_index, player]
        self.round_bets[self.round_index, player] += bet
        self.money[player] -= bet

    def _player_observation(self, player: int, rl_player: bool = False):
        return {
            'visible_cards_win_probability': np.array([self.winning_probabilities[player]], dtype='float32'),
            'money': np.array([self.money[player]], dtype='int32'),
            'players_playing': self.players_playing,
            'round_bets': self.round_bets,
            'player_ind': self.encoded_rl_id if rl_player else player,
        }

    def _play_non_rl_player(self, player_id: int):
        if not self.players_playing[player_id]:
            self.player_cycle.next_player()
            return
        if self.is_done:
            return
        player_observation = self._player_observation(player_id)
        self.apply_action(player_id, self.player_list[player_id].get_action(player_observation, self.action_space))
        self.player_cycle.next_player()
