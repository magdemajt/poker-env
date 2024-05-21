import ray
from ray.rllib.algorithms.ppo import PPOConfig
from player import RLPlayer, RandomPlayer
from ray.tune.logger import pretty_print
from poker_env import PokerEnv

ray.init()
config = PPOConfig()
# config.env_config = {"player_list": [RLPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]}
config.framework("torch")
config.num_workers = 1
config.num_gpus = 1
config.output = "ray_output"
config.train_batch_size = 128
algo = config.build(env=PokerEnv)
for i in range(10):
    result = algo.train()
    print(pretty_print(result))
    # if i % 100 == 0:
    #     checkpoint = algo.save()
    #     print("checkpoint saved at", checkpoint)
ray.shutdown()
