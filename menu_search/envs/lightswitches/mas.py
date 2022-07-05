from ray.rllib import MultiAgentEnv
from envs.lightswitches.environment import Environment
import numpy as np
from abc import ABC


class MultiAgentEnvironment(MultiAgentEnv, ABC):
    def __init__(self, env_config):
        super().__init__()
        self.low = env_config['low']
        self.mid = env_config['mid']
        _agent_ids = [
            "high_level_agent"
        ]
        for a in range(env_config['n_items']):
            _agent_ids.append(f"mid_level_agent_{a}")
            _agent_ids.append(f"low_level_agent_{a}")
        self._agent_ids = _agent_ids
        self.environment = Environment(env_config, self._agent_ids)
        self.n_episodes = 0
        self.steps_in_episode = 0
        self.cur_obs = None

    def reset(self) -> dict:
        self.n_episodes += 1
        self.steps_in_episode = 0
        if self.low == 'multi':
            self.low_level_agent_id = f"low_level_agent_{np.random.randint(self.environment.n_slots)}"
        else:
            self.low_level_agent_id = f"low_level_agent_0"

        if self.mid == 'single' or self.mid == 'heuristic':
            self.mid_level_agent_id = "mid_level_agent_0"
        elif self.mid == 'multi':
            self.mid_level_agent_id = f"mid_level_agent_{np.random.randint(self.environment.n_slots)}"
        obs = self.environment.reset()
        return {
            "high_level_agent": obs["high_level_agent"],
        }

    def step(self, action_dict: dict):
        agent = str(list(action_dict.keys())[0])
        if agent.startswith("high"):
            self.steps_in_episode += 1
            return self._high_level_step(action_dict["high_level_agent"])
        elif agent.startswith("mid"):
            return self._mid_level_step(action_dict[self.mid_level_agent_id])
        elif agent.startswith("low"):
            return self._low_level_step(action_dict[self.low_level_agent_id])

    def _high_level_step(self, action):
        f_obs, f_rew, f_done, add = self.environment.step_high_level(action)
        rew = {}
        if self.mid is not None:
            if self.mid == 'multi':
                self.mid_level_agent_id = f"mid_level_agent_{action}"
            else:
                self.mid_level_agent_id = f"mid_level_agent_0"

            obs = {self.mid_level_agent_id: f_obs[self.mid_level_agent_id]}

        else:
            if self.low == 'multi':
                self.low_level_agent_id = f"low_level_agent_{action}"
            else:
                self.low_level_agent_id = f"low_level_agent_0"
            obs = {self.low_level_agent_id: f_obs[self.low_level_agent_id]}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _mid_level_step(self, action):
        f_obs, f_rew, f_done, add = self.environment.step_mid_level(action)
        if self.low == 'multi':
            self.low_level_agent_id = f"low_level_agent_{action}"
        else:
            self.low_level_agent_id = f"low_level_agent_0"

        obs = {self.low_level_agent_id: f_obs[self.low_level_agent_id]}
        done = {"__all__": False}
        rew = {}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        f_obs, f_rew, f_done, _ = self.environment.step_low_level(action)
        rew = {
            "high_level_agent": f_rew['high_level_agent'],
            self.low_level_agent_id: f_rew['low_level_agent'],
        }

        if self.mid is not None:
            rew[self.mid_level_agent_id] = f_rew["mid_level_agent"]

        done = {"__all__": f_done}
        obs = {"high_level_agent": f_obs["high_level_agent"]}
        if isinstance(self.environment.diff, int):
            n_diff = self.environment.diff*2
        elif self.environment.diff == 'curriculum':
            n_diff = self.environment.task*2
        else:
            n_diff = self.environment.n_items * 2
        if self.steps_in_episode == max(5, n_diff):
            done["__all__"] = True
            rew = {}
            for a in self._agent_ids:
                rew[a] = - self.steps_in_episode
        return obs, rew, done, {}

    def set_task(self, task):
        self.environment.task = min(task, self.environment.n_items)

    def get_task(self):
        return self.environment.task