# Written by Mateusz Wejman
from __future__ import annotations
from enum import Enum
from typing import Tuple, Optional, List
import random

class Suit(Enum):
    Hearts = 1
    Spades = 2
    Diamonds = 3
    Clubs = 4


class CardValue(Enum):
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13

    def __lt__(self, other):
        if self == CardValue.Ace:
            return False
        if other == CardValue.Ace:
            return self != CardValue.Ace
        return self.value < other.value

    def __gt__(self, other):
        if self == CardValue.Ace:
            return other != CardValue.Ace
        if other == CardValue.Ace:
            return False
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __sub__(self, other):
        if isinstance(other, int):
            value = int(self.value)
            if value - other == 0:
                return CardValue.King
            return CardValue(value - other)

    def __add__(self, other):
        if isinstance(other, int):
            value = int(self.value)
            return CardValue((value + other) % 13 + 1)
    def __hash__(self):
        return self.value


class Card:
    suit: Suit
    value: CardValue

    def __init__(self, suit: Suit, value: CardValue):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __repr__(self):
        return f"{self.value} of {self.suit}"

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        if self.value == CardValue.Ace:
            return False
        if other.value == CardValue.Ace:
            return self.value != CardValue.Ace
        return self.value < other.value

    def __gt__(self, other):
        if self.value == CardValue.Ace:
            return other.value != CardValue.Ace
        if other.value == CardValue.Ace:
            return False
        return self.value > other.value


class HandStrength(Enum):
    RoyalFlush = 1
    StraightFlush = 2
    FourOfAKind = 3
    FullHouse = 4
    Flush = 5
    Straight = 6
    ThreeOfAKind = 7
    TwoPair = 8
    OnePair = 9
    HighCard = 10

    def __lt__(self, other):
        if other is None:
            return False
        if isinstance(other, int):
            return self.value < other
        if isinstance(other, HandStrength):
            return self.value < other.value
        raise ValueError("Cannot compare HandStrength to other type")

    def __gt__(self, other):
        if other is None:
            return True
        if isinstance(other, int):
            return self.value > other
        if isinstance(other, HandStrength):
            return self.value > other.value
        raise ValueError("Cannot compare HandStrength to other type")

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, HandStrength):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        raise ValueError("Cannot compare HandStrength to other type")


class Hand:
    cards: List[Card]  # 5 cards

    def __str__(self):
        return f"[ {self.cards[0]}\n {self.cards[1]}\n {self.cards[2]}\n {self.cards[3]}\n {self.cards[4]}\n ]"

    def __init__(self, cards: List[Card]):
        if len(cards) != 5:
            raise ValueError("Hand must have 5 cards")
        self.cards = cards

    def get_hand_strength_indicator(self):
        if len(self.cards) != 5:
            raise ValueError("Hand must have 5 cards")
        sorted_hand = sorted(self.cards)

        has_top_straight = (sorted_hand[4].value == CardValue.Ace
                            and sorted_hand[0].value == CardValue.Ten
                            and sorted_hand[1].value == CardValue.Jack
                            and sorted_hand[2].value == CardValue.Queen
                            and sorted_hand[3].value == CardValue.King)

        has_lowest_straight = (sorted_hand[4].value == CardValue.Ace
                            and sorted_hand[0].value == CardValue.Two
                            and sorted_hand[1].value == CardValue.Three
                            and sorted_hand[2].value == CardValue.Four
                            and sorted_hand[3].value == CardValue.Five)

        has_color = sorted_hand[0].suit == sorted_hand[1].suit \
                    and sorted_hand[1].suit == sorted_hand[2].suit \
                    and sorted_hand[2].suit == sorted_hand[3].suit \
                    and sorted_hand[3].suit == sorted_hand[4].suit

        if has_top_straight and has_color:
            return HandStrength.RoyalFlush, 14

        has_straight = has_top_straight or has_lowest_straight or (sorted_hand[0].value == sorted_hand[1].value - 1
                        and sorted_hand[1].value == sorted_hand[2].value - 1
                        and sorted_hand[2].value == sorted_hand[3].value - 1
                        and sorted_hand[3].value == sorted_hand[4].value - 1)




        if has_straight and has_color:
            return HandStrength.StraightFlush, max(sorted_hand, key=lambda x: x.value).value

        # check for four of a kind
        repeating = {}
        for card in sorted_hand:
            if card.value in repeating:
                repeating[card.value] += 1
            else:
                repeating[card.value] = 1

        to_return: [HandStrength, CardValue] = None, None

        sorted_repeating = sorted(repeating.items(), key=lambda x: x[1], reverse=True)

        for repeated_ind, repeated in sorted_repeating:
            if repeated == 4:
                return HandStrength.FourOfAKind, repeated_ind
            if repeated == 3:
                to_return = HandStrength.ThreeOfAKind, repeated_ind
            if repeated == 2 and to_return[0] == HandStrength.ThreeOfAKind:
                return HandStrength.FullHouse, to_return[1]
            elif repeated == 2 and to_return[0] == HandStrength.OnePair:
                to_return = HandStrength.TwoPair, max(to_return[1], repeated_ind)
            elif repeated == 2:
                to_return = HandStrength.OnePair, repeated_ind

        if to_return[0] == HandStrength.FullHouse:
            return to_return

        if has_color:
            return HandStrength.Flush, max(sorted_hand, key=lambda x: x.value).value

        if has_straight:
            return HandStrength.Straight, max(sorted_hand, key=lambda x: x.value).value

        if to_return[0] is not None:
            return to_return

        return HandStrength.HighCard, max(sorted_hand, key=lambda x: x.value).value

        # the stronger the hand the lower the indicator
        # 1. Royal Flush
        # 2. Straight Flush
        # 3. Four of a Kind
        # 4. Full House
        # 5. Flush
        # 6. Straight
        # 7. Three of a Kind
        # 8. Two Pair
        # 9. One Pair
        # 10. High Card

    def __eq__(self, other):
        self_strength, self_indicator = self.get_hand_strength_indicator()
        other_strength, other_indicator = other.get_hand_strength_indicator()

        if self_strength != other_strength:
            return False

        if self_indicator != other_indicator:
            return False

        if self_strength == HandStrength.FourOfAKind or self_strength == HandStrength.TwoPair \
                or self_strength == HandStrength.OnePair or self_strength == HandStrength.ThreeOfAKind:
            the_kind = self_indicator
            self_kicker = max([card for card in self.cards if card.value != the_kind])
            other_kicker = max([card for card in other.cards if card.value != the_kind])
            return self_kicker == other_kicker

        return True

    def __gt__(self, other):
        self_strength, self_indicator = self.get_hand_strength_indicator()
        other_strength, other_indicator = other.get_hand_strength_indicator()

        if self_strength < other_strength:
            return True

        if self_strength > other_strength:
            return False

        if self_indicator < other_indicator:
            return True

        if self_strength == HandStrength.FourOfAKind or self_strength == HandStrength.TwoPair or self_strength == HandStrength.OnePair or self_strength == HandStrength.ThreeOfAKind:
            the_kind = self_indicator
            self_kicker = max([card for card in self.cards if card.value != the_kind])
            other_kicker = max([card for card in other.cards if card.value != the_kind])
            return self_kicker > other_kicker

        return False


class Deck:
    cards: List[Card]
    drawn_index: int
    def __init__(self):
        self.cards = [Card(suit, value) for suit in Suit for value in CardValue]
    def shuffle(self):
        random.shuffle(self.cards)
        self.drawn_index = 0
    def burn_n(self, n: int):
        self.drawn_index += n
    def draw_n(self, n: int):
        self.drawn_index += n
        if self.drawn_index > len(self.cards):
            raise ValueError("Not enough cards in the deck")
        return self.cards[self.drawn_index - n:self.drawn_index]
