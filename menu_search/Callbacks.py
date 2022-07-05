from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict
import numpy as np


class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.current_level = 0
        self.last_act_low = [0, 0, 0]
        self.training_count = 0
        self.additional_metrics = {}

    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        self.additional_metrics = {}

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        additional_metrics = worker.env.environment.add
        if additional_metrics['low_level'] == 1:
            for key in additional_metrics:
                if key in self.additional_metrics:
                    self.additional_metrics[key].append(additional_metrics[key])
                else:
                    self.additional_metrics[key] = [additional_metrics[key]]

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        for key in self.additional_metrics:
            episode.custom_metrics[key] = np.mean(self.additional_metrics[key])

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        def foreach_worker(ev):
            def foreach_env(env):
                if result[
                    "episode_len_mean"] < 3.1 * env.environment.task and env.environment.diff == 'curriculum' and env.environment.since_last_update > 50:
                    env.set_task(env.get_task() + 1)
                    env.environment.since_last_update = 0
                else:
                    env.environment.since_last_update += 1

            ev.foreach_env(foreach_env)

        trainer.workers.foreach_worker(
            foreach_worker
        )
