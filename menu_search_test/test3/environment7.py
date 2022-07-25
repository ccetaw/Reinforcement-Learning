from abc import ABC
from cmath import pi
import gym
from gym import spaces
import numpy as np
from pandas import array
from env_utils import (
    check_within_element,
    generate_item_to_slot,
    generate_goal,
    calc_distance,
    clip_normalize_reward,
    compute_stochastic_position,
    scale,
    compute_width_distance,
    who_mt,
    compute_width_distance_fast
)
from interface import Interface


"""
Setting:
    Small steps
"""


class Environment(gym.Env, ABC, Interface):

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}
    reward_range = (-float("inf"), float("inf"))
    
    def __init__(self, envconfig):
        super().__init__(envconfig)
        self.dt = 0.01  # seconds

        self.action_space = spaces.Box(low=0., high=2 * np.pi, shape=(1,), dtype=np.float64)
        obs_space_dict = {
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'target': spaces.Box(low=0., high=1., shape=(2,))
        } 
        self.observation_space = spaces.Dict(obs_space_dict)

        self.done = False
        self.counter = 0
        self.counter_max = 0
        self.state = {
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'target': spaces.Box(low=0., high=1., shape=(2,)),
            'velocity': spaces.Box(-1/self.dt, 1/self.dt, shape=(2,))
        }
        self.add = {} # everything else

        self.screen = None
        self.clock = None
        self.isopen = True

    def get_state(self):
        obs = {}
        for key in self.observation_space:
            obs[key] = self.state[key]
        return obs 

    def get_reward(self, action):
        reward = 0
        direction = np.array([np.cos(action), np.sin(action)])
        desired_direction = self.state['velocity'] / calc_distance(self.state['velocity'], np.array([0,0]), method= 'l1')
        reward += (np.dot(direction, desired_direction) - 1)
        if 0<= self.state['cursor_position'][0] <= 1 and 0<= self.state['cursor_position'][1] <= 1:
            pass
        else:
            reward -= 3
            self.state['cursor_position'] = np.clip(self.state['cursor_position'], 0, 1)
        reward -= calc_distance(self.state['cursor_position'], self.state['target'], 'l1')

        if calc_distance(self.state['cursor_position'], self.state['target'], 'l1') < 4*calc_distance(self.state['velocity'] * self.dt, np.zeros(2), 'l1'):
            self.done = True
            reward += 100
        elif self.counter >= 250:
            self.done = True
            reward -= 50
        return reward

    def step(self, action):
        _action = action[0]
        self.state['cursor_position'] += calc_distance(self.state['velocity'], np.array([0,0]), method= 'l1')  * np.array([np.cos(_action), np.sin(_action)]) * self.dt
        self.counter += 1
        reward = self.get_reward(_action)
        return self.get_state(), reward, self.done, {}

    def reset(self):
        self.state['target'] = np.random.random_sample(size=(2,))
        self.state['cursor_position'] = np.random.random_sample(size=(2,))
        _, movetime = compute_stochastic_position(self.state['target'],self.state['cursor_position'], sigma=0.01)
        self.state['velocity'] = (self.state['target'] - self.state['cursor_position']) / movetime
        self.counter_max = int(movetime / self.dt)

        self.done = False
        self.counter = 0
        return self.get_state()

    def render(self, mode="human"):
        import pygame
        from pygame import gfxdraw

        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))
        x, y = self.state['cursor_position'][0] * self.screen_width, self.state['cursor_position'][1] * self.screen_height
        gfxdraw.filled_circle(surf, int(x), int(y), 20, (0, 0, 255))
        x, y = self.state['target'][0] * self.screen_width, self.state['target'][1] * self.screen_height
        gfxdraw.circle(surf, int(x), int(y), 20, (255, 0, 0))

        self.screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        import pygame
        pygame.display.quit()
        pygame.quit()
        self.isopen = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    args = parser.parse_args()

    env_config = {
        'random':False,
        'n_items':7,
    }
    
    if args.demo:
        env = Environment(env_config)
        for i in range(3):
            env.reset()
            while not env.done:
                env.step(env.action_space.sample())
                env.render()
        env.close()

    if args.dry_train:
        from ray.rllib.agents import ppo
        from ray.tune.logger import pretty_print
        config = {
            "env": Environment,  # or "corridor" if registered above
            "env_config": env_config,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "gamma": 0.9
        }

        stop = {
            "training_iteration": 500,
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

        trainer.save("test3_7")

        env = Environment(env_config)
        for i in range(10):
            env.reset()
            while not env.done:
                action = trainer.compute_single_action(env.get_state())
                env.step(action)
                env.render()
        env.close()

        