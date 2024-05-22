import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from player import RLPlayer, RandomPlayer
from ray.tune.logger import pretty_print
from poker_env import PokerEnv
from time import time


def train_from_checkpoint(checkpoint_path: str, num_epochs: int, experiment_name: str = "ray_output"):
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    algo = Algorithm.from_checkpoint(checkpoint_path)
    for i in range(num_epochs):
        result = algo.train()
        print("epoch ", i, ", result ", result["episode_reward_mean"])
        if i % 10 == 0:
            time_str = str(time())
            checkpoint = algo.save(f"{experiment_name}/checkpoint{i}_{time_str}")
            print("checkpoint saved at", checkpoint)
            print(pretty_print(result))

    ray.shutdown()


def train_new(num_epochs: int, experiment_name: str = "ray_output"):
    player_list = [RLPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    config = PPOConfig()
    config.framework("torch")
    config.num_workers = 1
    config.num_gpus = 1
    config.output = experiment_name
    config.train_batch_size = 2048
    config.environment(PokerEnv, env_config={"player_list": player_list})
    algo = config.build()
    for i in range(num_epochs):
        result = algo.train()
        print("epoch ", i, ", result ", result["episode_reward_mean"])
        if i % 10 == 0:
            time_str = str(time())
            checkpoint = algo.save(f"{experiment_name}/checkpoint{i}_{time_str}")
            print("checkpoint saved at", checkpoint)
            print(pretty_print(result))

    ray.shutdown()


if __name__ == '__main__':
    train_new(10, "player_list")
    # train_from_checkpoint("ray_output/checkpoint0_1633940004.3669786", 10)
