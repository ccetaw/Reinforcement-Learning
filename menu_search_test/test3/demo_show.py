from turtle import Turtle
import ray
from environment7 import Environment
from ray.rllib.agents import ppo
from ray import tune

env_config = {
        'random': False,
        'n_items': 10,
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
agent.restore('menu_search_test/test3/test3_7/checkpoint_000500/checkpoint-500')

env = Environment(env_config)
for i in range(5):  
    env.reset()
    while not env.done:
        env.render()
        action = agent.compute_single_action(env.get_state(), unsquash_action=True)
        env.step(action)
env.close()