from turtle import Turtle
import ray
from environment import Environment
from ray.rllib.agents import ppo
from ray import tune

env_config = {
        'random': False,
        'n_items': 10,
        'num_targets': 2
    }
config = {
    "env": Environment,  # or "corridor" if registered above
    "env_config": env_config,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_workers": 1,  # parallelism
    "framework": "torch",
    "gamma": 0.9,
    "log_level": "CRITICAL",
    # "callbacks": CustomMetricsCallback,
    "seed": 31415,
}

config = {**ppo.DEFAULT_CONFIG, **config}
agent = ppo.PPOTrainer(config=config)
agent.restore('/home/cetaceanw/repo/Reinforcement-Learning/ray_results/PPOTrainer_2022-07-12_16-32-42/7d024_00013-random_False-n_items_10-num_targets_2_13_n_items=10,num_targets=2,random=False_2022-07-12_16-47-37/checkpoint_000042/checkpoint-42')

env = Environment(env_config)
for i in range(5):
    env.reset()
    while not env.done:
        env.render()
        action = agent.compute_single_action(env.state, unsquash_action=True)
        env.step(action)
env.close()