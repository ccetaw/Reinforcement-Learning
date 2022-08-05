from env import ButtonPanel
from ray.rllib.agents import ppo
import numpy as np
from tqdm import tqdm



env_config = {
    'random': True,
    'n_buttons': 6,
    'mode': 'all_exclu'
}

config = {
    "num_workers": 3,
    "env": ButtonPanel,
    "env_config": env_config,
    "gamma": 0.9,
    "framework": "torch",
    "log_level": "CRITICAL",
    "num_gpus": 0,
    "seed": 31415,
}

config = {**ppo.DEFAULT_CONFIG, **config}
agent = ppo.PPOTrainer(config=config)
agent.restore('./trained_user/PPOTrainer_2022-08-01_20-04-46/random_True-n_buttons_6-save_ui_True-mode_all_exclu/checkpoint_000200/checkpoint-200')
env = ButtonPanel(env_config)

env.save('./sa_1')