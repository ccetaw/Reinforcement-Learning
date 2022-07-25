import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from env import MultiAgentUI
from utils import (policy_mapping_fn, trial_name_string)


def generate_multiagent_dictionary(spec):
    env = MultiAgentUI(spec.config.env_config)
    policies = {
        "user_high": (
            None,
            env.observation_space['user_high'],
            env.action_space['user_high'],
            {}
        ),
        "user_low":(
            None,
            env.observation_space['user_low'],
            env.action_space['user_low'],
            {}
        )   
    }
    return {"policies": policies,
            "policy_mapping_fn": policy_mapping_fn}

ray.init(local_mode=False, num_cpus=24, num_gpus=0)

env_config = {
    'random': tune.grid_search([True, False]),
    'n_buttons': tune.grid_search([7, 10, 15]),
    'save_ui': tune.grid_search([True])
}

stop = {
    "training_iteration": 1000,
}
config = {
    "num_workers": 3,
    "env": MultiAgentUI,
    "env_config": env_config,
    "gamma": 0.9,
    "multiagent": tune.sample_from(generate_multiagent_dictionary),
    "framework": "torch",
    "log_level": "CRITICAL",
    "num_gpus": 0,
    "seed": 31415,
}

config = {**DEFAULT_CONFIG, **config}
results = tune.run(
    PPOTrainer,
    trial_name_creator=trial_name_string,
    stop=stop,
    config=config,
    local_dir="ray_results",
    verbose=1,
    checkpoint_freq=50,
    checkpoint_at_end=True,
    num_samples=1,
    callbacks=[
        WandbLoggerCallback(
                project=f"menu_search_test",
                entity="cetaceanw",
                group=f"simple_hierachical",
                job_type="train",
                log_config=True,
                save_code=True,
        )
    ],
)
ray.shutdown()