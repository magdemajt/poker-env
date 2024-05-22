import ray
from ray.rllib.algorithms.ppo import PPOConfig
from player import RLPlayer, RandomPlayer
from ray.tune.logger import pretty_print
from poker_env import PokerEnv

if __name__ == '__main__':

    ray.init()
    config = PPOConfig()
    # config.env_config = {"player_list": [RLPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]}
    config.framework("torch")
    config.num_workers = 1
    config.num_gpus = 1
    config.output = "ray_output"
    config.train_batch_size = 128
    config.environment(PokerEnv)
    algo = config.build()
    for i in range(1000):
        result = algo.train()
        if i % 10 == 0:
            # checkpoint = algo.save()
            print("epoch ", i, ", result ", result["episode_reward_mean"])
            # print("checkpoint saved at", checkpoint)
    ray.shutdown()
