from poker_lib import get_chances, get_win_indices


# get_chances
cards = ["2s", "ts", "4s", "js", "qs", "ks", "as"]
total_players = 2
iterations = 1000

for i in [2, 5, 6, 7]:
    try:
        result = get_chances(cards[:i], total_players, iterations)
        print(f"Chances with {i} cards: {result}")
    except Exception as e:
        print(e)


# get_winner_index
table = ["2s", "ts", "4s", "js", "qh"]
players_1 = [["3c", "3d"], ["2c", "2d"]]
players_2 = players_1[::-1]
winner_index_1 = get_win_indices(table, players_1)
winner_index_2 = get_win_indices(table, players_2)
print(f"Winner index 1: {winner_index_1}")
print(f"Winner index 2: {winner_index_2}")
