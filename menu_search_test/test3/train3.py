import ray
from environment2 import Environment
from ray.rllib.agents import ppo
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from train_util import trial_name_string


ray.init(local_mode=False, num_cpus=8, num_gpus=0)
env_config = {
        'random':tune.grid_search([True, False]),
        'n_items':tune.grid_search([7, 12]),
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
    "seed": 31415,
}

stop = {
    "training_iteration": 1000,
    "episode_reward_mean": -1,
}

config = {**ppo.DEFAULT_CONFIG, **config}
results = tune.run(
        ppo.PPOTrainer,
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
                group=f"small_steps",
                job_type="train",
                log_config=True,
                save_code=True,
            )
        ],
    )

ray.shutdown()