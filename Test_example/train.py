import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from Callbacks import CustomMetricsCallback

from envs.lightswitches.mas import (
    MultiAgentEnvironment as MASEnvironment,
)
from utils.train_util import trial_name_string, policy_mapping_fn

if __name__ == "__main__":
    ray.init(local_mode=False, num_cpus=35, num_gpus=0)

    env_config = {
        'multi': tune.grid_search([True, False]),
        'mid': tune.grid_search(["multi", "single", None]),
        'random': tune.grid_search([True, False]),
        'n_targets': tune.grid_search([5]),
        'n_items': tune.grid_search([5]),
        'n_diff': tune.grid_search([1, 3, 5])
    }

    env = MASEnvironment(env_config, init=True)
    policies = {
        "high_level_agent_policy": (
            None,
            env.environment.observation_space_user_high,
            env.environment.action_space_high,
            {}
        )
    }
    for a in env._agent_ids:
        if a.startswith("low"):
            policies[f"{a}_policy"] = (
                None,
                env.environment.observation_space_user_low,
                env.environment.action_space_low,
                {},
            )
        elif a.startswith("mid"):
            policies[f"{a}_policy"] = (
                None,
                env.environment.observation_space_user_mid,
                env.environment.action_space_mid,
                {},
            )
    stop = {
        "training_iteration": 1000,
    }
    config = {
        "num_workers": 3,
        "env": MASEnvironment,
        "env_config": env_config,
        "gamma": 0.9,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "torch",
        "log_level": "CRITICAL",
        "callbacks": CustomMetricsCallback,
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
                project=f"RL_test",
                entity="cetaceanw",
                group=f"lightswitch_1",
                job_type="train",
                log_config=True,
                save_code=True,
            )
        ],
    )
    ray.shutdown()
