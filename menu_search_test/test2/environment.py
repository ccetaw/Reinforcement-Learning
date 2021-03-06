from abc import ABC
import gym
from gym import spaces
import numpy as np
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
import pygame
from pygame import gfxdraw

import argparse
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--demo", action="store_true", help="Show a window of the system")
group.add_argument("--dry_train", action="store_true", help="Train the agent")
args = parser.parse_args()


class Environment(gym.Env, ABC, Interface):
    """
    Setting:
        We have multiple slots and multiple targets. We want the curosr to hit the targets in whatever order. The agent takes a start_action and the actions before arriving at the start_action
        will be deterministic. Actions generated by the policy different from the start_action will be penalized thus the agent will learn to focus. The start_action is penalized by a reward 
        function that is zero around the target and nonzero everywhere else. Since the steps after start_action are deterministic, no time penalty will be added.
    Details
        - If the cursor hit the target before arriving at start_action, the cursor continues to the start_action, since the start_action will be moved to a higher hierachy agent afterwards
        - After the cursor arrives at the target, it will stop moving for one step to 
            - generate a new start_action
            - simulate the human behavior
    """

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 60}
    reward_range = (-float("inf"), float("inf"))
    
    def __init__(self, envconfig):
        interface_config = {
            'n_items': envconfig['n_items'],
            'random': envconfig['random']
        }
        super().__init__(interface_config)
        self.num_targets = envconfig['n_targets']
        self.dt = 0.01  # seconds

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
        )
        obs_space_dict = {
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'target_slots': spaces.MultiBinary(self.n_slots),
            'menu': spaces.Box(low=np.array([0.0, 0.0] * self.n_slots), high=np.array([1.0, 1.0] * self.n_slots), dtype=np.float64),
            'is_moving': spaces.MultiBinary(1),
            'selected_slots': spaces.MultiBinary(self.n_slots),
            'start_action': self.action_space,
        } 

        self.state = {
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'menu': spaces.Box(low=np.array([0.0, 0.0] * self.n_slots), high=np.array([1.0, 1.0] * self.n_slots), dtype=np.float64),
            'target_slots': spaces.MultiBinary(self.n_slots),
            'is_moving': spaces.MultiBinary(1),
            'selected_slots': spaces.MultiBinary(self.n_slots),
            'last_action': self.action_space,
            'start_action': self.action_space,
            'velocity': spaces.Box(-1/self.dt, 1/self.dt, shape=(2,))
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        self.done = False
        self.counter = 0
        self.screen = None
        self.clock = None
        self.isopen = True
    
    def get_state(self):
        obs = {}
        for key in self.observation_space:
            obs[key] = self.state[key]
        return obs 

    def get_start_action_reward(self):
        targets = np.where(self.state['target_slots'] == 1)[0]
        reward = 1
        for target in targets:
            target_center = np.asarray(
                [self.ui[target]['x'] + self.ui[target]['w'] / 2, self.ui[target]['y'] + self.ui[target]['h'] / 2])
            if self.state['selected_slots'][target] == 0:
                reward *= calc_distance(self.state['cursor_position'], target_center,'l2')
                if check_within_element(self.state['cursor_position'], self.ui[target]):
                    reward = 0

        return -reward

    def step(self, action, test=False):
        reward = 0
        _action = np.asarray(action, dtype=np.float64)
        if np.array_equal(self.state['is_moving'], np.array([0])): # If the cursor is not moving, start moving
            self.state['is_moving'] = np.array([1])
            start_action, move_time = compute_stochastic_position(_action, self.state['cursor_position'], sigma=0.01)
            self.state['start_action'] = _action
            self.state['velocity'] = (self.state['start_action'] - self.state['cursor_position'])/move_time  
            reward += self.get_start_action_reward()
            self.counter += 1
        else:
            reward += -calc_distance(self.state['start_action'], action, 'l2')
        reward += self.get_start_action_reward() 
        if test:
            _, move_time = compute_stochastic_position(_action, self.state['cursor_position'], sigma=0.01)
            velocity = (_action - self.state['cursor_position'])/move_time  
            self.state['cursor_position'] += velocity * self.dt
        else:
            self.state['cursor_position'] += self.state['velocity'] * self.dt

        if calc_distance(self.state['cursor_position'], self.state['start_action'], 'l2') < calc_distance(self.state['velocity'] * self.dt, np.zeros(2), 'l2'):
            is_arrived = True 
            self.state['is_moving'] = np.array([0])
        else:
            is_arrived = False

        if is_arrived:
            for slot in self.ui.keys():
                if check_within_element(self.state['cursor_position'], self.ui[slot]) and self.state['target_slots'][slot] == 1 :
                    self.state['selected_slots'][slot] = 1
                    reward += 1

        if np.array_equal(self.state['target_slots'], self.state['selected_slots']):
            self.done = True
        elif self.counter > 10:
            self.done = True
            reward += -5

        return self.get_state(), reward, self.done, {}

    def reset(self):
        slot = np.zeros(self.n_slots)
        idx = np.random.randint(low=0, high=self.n_slots, size=self.num_targets)
        while len(set(idx)) != self.num_targets: # Make sure we have num_targets different targets
            idx = np.random.randint(low=0, high=self.n_slots, size=self.num_targets)
        slot[idx] =  1
        self.state['target_slots'] = slot
        self.state['cursor_position'] = np.random.random_sample(size=(2,))
        self.state['is_moving'] = np.array([0])
        # self.state['last_action'] = None
        self.state['selected_slots'] = np.zeros(self.n_slots)
        self.state['start_action'] = np.zeros(2)
        self.state['velocity'] = np.zeros(2)
        if self.random:
            self.generate_random_ui()
        else:
            self.generate_static_ui()
        self.state['menu'] = self.get_menu_location()

        self.done = False
        self.counter = 0
        return self.get_state()

    def render(self, mode="human"):
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
        font = pygame.font.Font('freesansbold.ttf', 16)
        x, y = self.state['cursor_position'][0] * self.screen_width, self.state['cursor_position'][1] * self.screen_height
        targets = np.where(self.state['target_slots'] == 1)[0]
        for target in targets:
            if check_within_element(self.state["cursor_position"], self.ui[target]):
                gfxdraw.filled_circle(surf, int(x), int(y), 20, (0, 0, 255))
            else:
                gfxdraw.filled_circle(surf, int(x), int(y), 20, (255, 0, 0))
        
        if self.state['start_action'] is not None:
            x, y = self.state['start_action'][0] * self.screen_width, self.state['start_action'][1] * self.screen_height
            gfxdraw.circle(surf, int(x), int(y), 20, (0, 150, 150))

        for element in self.ui:
            rect = pygame.Rect(self.ui[element]['x'] * self.screen_width,
                               self.ui[element]['y'] * self.screen_height,
                               self.ui[element]['w'] * self.screen_width,
                               self.ui[element]['h'] * self.screen_height)
            if element in targets:
                if self.state['selected_slots'][element] == 1:
                    gfxdraw.rectangle(surf, rect, (200, 200, 0))
                else:
                    gfxdraw.rectangle(surf, rect, (200, 0, 0))
            else:
                gfxdraw.box(surf, rect, (100, 100, 100))

        self.screen.blit(surf, (0, 0))
        for element in self.ui:
            t = f"Slot:{element} I:{self.ui[element]['item']}"
            text = font.render(t, True, (0, 0, 0))
            x = int((self.ui[element]['x'] + self.ui[element]['w'] / 2) * self.screen_width)
            y = int((self.ui[element]['y'] + self.ui[element]['h'] / 2) * self.screen_height)
            self.screen.blit(text, (x, y))

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
    env_config = {
        'random':False,
        'n_items':10,
        'n_targets': 3
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
        trainer.save()

        env = Environment(env_config)
        for i in range(5):
            env.reset()
            while not env.done:
                action = trainer.compute_single_action(env.state)
                env.step(action, test=False)
                env.render()
        env.close()

        