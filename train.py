import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

from model import MODEL_DEFAULTS
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
            reward = result["episode_reward_mean"]
            checkpoint = algo.save(f"{experiment_name}/checkpoint{i}_{time_str}_rew_{reward}")
            print("checkpoint saved at", checkpoint)
            print(pretty_print(result))

    ray.shutdown()


def train_new(num_epochs: int, experiment_name: str = "ray_output"):
    player_list = [RLPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]
    if ray.is_initialized():
        ray.shutdown()
    # config https://docs.ray.io/en/latest/rllib/rllib-models.html#default-model-config-settings

    ray.init()
    config = PPOConfig()
    config.num_workers = 1
    config.num_gpus = 1
    config.output = experiment_name
    config.train_batch_size = 2048
    config.environment(PokerEnv, env_config={"player_list": player_list})
    config.output = "ray_output"
    config.train_batch_size = 128
    config.environment(PokerEnv)
    updated = MODEL_DEFAULTS.copy()
    MODEL_DEFAULTS.update({

        # "custom_model": "my_model",
        "fcnet_hiddens": [56, 128, 256, 128, 16],
        "fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "_disable_action_flattening": False,
        "_disable_preprocessor_api": False,
        "max_seq_len": 20
    })
    config.model = updated
    config.framework("torch")
    algo = config.build()
    for i in range(num_epochs):
        result = algo.train()
        print("epoch ", i, ", result ", result["episode_reward_mean"])
        if (i + 1) % 10 == 0:
            time_str = str(time())
            reward = result["episode_reward_mean"]
            checkpoint = algo.save(f"{experiment_name}/checkpoint{i}_{time_str}_rew_{reward}")
            print("checkpoint saved at", checkpoint)
            print(pretty_print(result))

    ray.shutdown()


if __name__ == '__main__':
    train_new(500, "outputs/relu_model")
    # train_from_checkpoint("ray_output/checkpoint0_1633940004.3669786", 10)
