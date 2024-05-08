# Poker Hand Evaluator

## Description

This library is a simple poker hand evaluator. It takes a list of cards and returns the best hand that can be made with those cards.

## Prerequisites

- rust (cargo)
- python3
- maturin (can be installed with pipx or pip)

## Installation

Run the following command to install the package in your current python environment:

```bash
maturin develop --release
```

## Usage

### get_chances

```python
>>> from poker_lib import get_chances
>>> 
>>> cards = ["2H", "3H", "4H", "5H", "6H", "7H", "8H"]
>>> number_of_players = 2
>>> iterations = 1000
>>> 
>>> chance_to_win = get_chances(cards, number_of_players, iterations)
>>> print(chance_to_win)
0.0


```

You can find this code in `example.py`

- The list of cards must have 2, 5, 6 or 7 cards. Cards in this list must be ordered as follows:

  - [0,1] - cards of the agent
  - [2,3,4,5,6] - cards exposed on table.

- Number of players is a total number of players. Our agent + n opponents -> n+1 players. E.g. Our agent and 2 opponents -> 3 players.
- Number of iterations - number of randomized games played

### get_win_indices

```python
>>> from poker_lib import get_win_indices
>>>
>>> table_cards = ["2H", "3H", "4H", "5H", "6H"]
>>> agent_1_cards = ["7H", "8H"]
>>> agent_2_cards = ["9H", "TH"]
>>> indices = get_win_indices(table_cards, [agent_1_cards, agent_2_cards])
>>> print(indices)
[0, 1]
```

### Argument notes

- Cards are represented as strings with the following format: `"<RANK><SUIT>"`. The rank is one of: `[2, 3, 4, 5, 6, 7, 8, 9, t, j, q, k ,a]`
 and the suit is a letter from `H` (hearts), `D` (diamonds), `C` (clubs) and `S` (spades).

## TODO

function which gets cards on table, cards of each agent and returns index of the winner.
