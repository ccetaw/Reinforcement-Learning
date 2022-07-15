from environment import Environment
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print


config = {
    "env": Environment,  # or "corridor" if registered above
    "env_config": {
        'low': 'coordinate',  # True, False, 'heuristic', 'coordinate'
        'mid': 'None',  # None, 'multi', 'single', 'heuristic'
        'random':False,
        'relative': False,
        'n_items':7,
        'n_diff': 'curriculum'
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_workers": 1,  # parallelism
    "framework": "torch",
    "gamma": 0.9
}

stop = {
    "training_iteration": 150,
    "timesteps_total": 100000,
    "episode_reward_mean": -1,
}

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)
# use fixed learning rate instead of grid search (needs tune)
ppo_config["lr"] = 1e-3
trainer = ppo.PPOTrainer(config=ppo_config, env=Environment)
# run manual training loop and print results after each iteration
for _ in range(stop["training_iteration"]):
    result = trainer.train()
    print(pretty_print(result))
    # stop training of the target train steps or reward are reached
    if (
        result["timesteps_total"] >= stop["timesteps_total"]
        or result["episode_reward_mean"] >= stop["episode_reward_mean"]
    ):
        break