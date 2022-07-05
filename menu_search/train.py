import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from Callbacks import CustomMetricsCallback
from envs.lightswitches.mas import (
    MultiAgentEnvironment as MASEnvironment,
)
from utils.train_util import trial_name_string, policy_mapping_fn
import numpy as np
from gym import spaces


def generate_multiagent_dictionary(spec):
    env = MASEnvironment(spec.config.env_config)
    policies = {}
    full_obs_dict = {
        "position": spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64),
        "item_to_slot": spaces.MultiBinary(env.environment.n_items * env.environment.n_slots),
        "menu": spaces.Box(low=0.0, high=1.0, dtype=np.float64, shape=(2 * env.environment.n_slots,)),
        "item_target": spaces.MultiBinary(env.environment.n_slots),
        "selection": spaces.MultiBinary(env.environment.n_items),
        "goal": spaces.MultiBinary(env.environment.n_items),
        "difference": spaces.MultiBinary(env.environment.n_items),
        'target_coordinate': spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64),
    }
    if env.environment.low == 'coordinate':
        full_obs_dict["slot_target"] = spaces.Box(low=np.array([0.0, 0.0]),
                                                  high=np.array([1.0, 1.0]),
                                                  dtype=np.float64)
    else:
        full_obs_dict["slot_target"] = spaces.MultiBinary(env.environment.n_slots)

    for a in env._agent_ids:
        obs = {}
        act = {}
        if a.startswith("high"):
            for key in env.environment.observation_space_high:
                obs[key] = full_obs_dict[key]
            if env.environment.mid is not None:
                act = spaces.Discrete(env.environment.n_items)
            else:
                act = spaces.Discrete(env.environment.n_slots)
        elif a.startswith("mid"):
            for key in env.environment.observation_space_mid:
                obs[key] = full_obs_dict[key]
            act = spaces.Discrete(env.environment.n_slots)

        elif a.startswith("low"):
            for key in env.environment.observation_space_low:
                obs[key] = full_obs_dict[key]
            act = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float64,
            )
        policies[a] = (None, spaces.Dict(obs), act, {})
    env.close()
    return {"policies": policies,
            "policy_mapping_fn": policy_mapping_fn}


if __name__ == "__main__":
    ray.init(local_mode=False, num_cpus=25, num_gpus=0)
    env_config = {
        'low': tune.grid_search(['coordinate']),  # True, False, 'heuristic', 'coordinate'
        'mid': tune.grid_search(['heuristic']),  # None, 'multi', 'single', 'heuristic'
        'random': tune.grid_search([True]),
        'relative': tune.grid_search([False]),
        'n_items': tune.grid_search([20, 15, 10]),
        'n_diff': tune.grid_search([5, 7, 'random', 'curriculum']),
    }

    stop = {
        "training_iteration": 1000,
    }
    config = {
        "num_workers": 1,
        "env": MASEnvironment,
        "env_config": env_config,
        "gamma": 0.9,
        "multiagent": tune.sample_from(generate_multiagent_dictionary),
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
                project=f"MARLUI2.1",
                entity="cetaceanw",
                group=f"run1",
                job_type="train",
                log_config=True,
                save_code=True,
            )
        ],
    )
    ray.shutdown()
