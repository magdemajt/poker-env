from poker_lib import get_chances

cards = ["2s", "ts", "4s", "js", "qs", "ks", "as"]
total_players = 2
iterations = 1000

for i in [2, 5, 6, 7]:
    try:
        result = get_chances(cards[:i], total_players, iterations)
        print(result)
    except Exception as e:
        print(e)
