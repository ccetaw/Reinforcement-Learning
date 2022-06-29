from abc import ABC
import gym
from gym import spaces
import numpy as np
from utils.env_utils import (
    check_within_element,
    generate_item_to_slot,
    generate_goal,
    calc_distance,
    clip_normalize_reward
)
from interface import Interface
import copy
import glob
import os
import time


class Environment(gym.Env, ABC, Interface):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_config, agent_ids):
        super().__init__(env_config)
        self.action_space_low = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
        ) # A class. R^d (bounded) box||   Main method .sample() Generates a single random sample inside the Box.
        self.action_space_mid = spaces.Discrete(self.n_targets) # A space consisting of finitely many elements.
        self.mid = env_config['mid'] # a list ["multi", "single", "None"]

        self.n_goal_diff = env_config['n_diff']
        if isinstance(self.n_goal_diff, dict):
            self.n_goal_diff = 1


        if self.mid is not None:
            self.action_space_high = spaces.Discrete(self.n_items)
        else:
            self.action_space_high = spaces.Discrete(self.n_targets)

        self.full_dict = {
            "position": spaces.Box( # Cursor Postion
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64
            ),
            "item_to_slot": spaces.MultiBinary( # Flatten look-up table of items-to-slots assignment
                self.n_items * self.n_targets # = 7*7  ??? n_target = n_slot ???
            ),
            "menu": spaces.Box(
                low=np.array([0.0] * self.n_targets), high=np.array([1.0] * self.n_targets), dtype=np.float64
            ),
            "item_target": spaces.MultiBinary(self.n_items),
            "slot_target": spaces.MultiBinary(self.n_targets),
            "selection": spaces.MultiBinary(self.n_items),
            "goal": spaces.MultiBinary(self.n_items), # There can be several goals?
            "difference": spaces.MultiBinary(self.n_items),
        }

        self.observation_dict_user_high = {
            "position": spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64
            ),
            "item_to_slot": spaces.MultiBinary(
                self.n_items * self.n_targets
            ),
            "selection": spaces.MultiBinary(self.n_items),
            "goal": spaces.MultiBinary(self.n_items),
            "difference": spaces.MultiBinary(self.n_items),
        }

        self.observation_dict_user_mid = {
            "position": spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64
            ),
            "item_to_slot": spaces.MultiBinary(
                self.n_items * self.n_targets
            ),
            "item_target": spaces.MultiBinary(self.n_items),
        }

        self.observation_dict_user_low = {
            "slot_target": spaces.MultiBinary(self.n_targets),
            "menu": spaces.Box( 
                low=np.array([0.0] * 2), high=np.array([1.0] * 2), dtype=np.float64
            ),  # Different from the one defined in full_dict
            "position": spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64
            ),
        }

        self.observation_space_user_high = spaces.Dict(
            self.observation_dict_user_high
        )
        self.observation_space_user_mid = spaces.Dict(
            self.observation_dict_user_mid
        )
        self.observation_space_user_low = spaces.Dict(
            self.observation_dict_user_low
        )
        self.add = {}  # whatever additional information there is
        self.state = None # Will be a dictionary: {"postion": x, "goal": x, "selection": x, }
        self.done = False
        self.lower_level_step_counter = 0
        self.ep_counter = 0
        self.old_state, self.actions, self.new_state, self.rewards = [], [], [], []
        self.screen = None
        self.clock = None
        self.isopen = True
        self.random = env_config['random'] # \in [True, False]
        if not self.random:
            self.generate_static_ui()

        self._agent_ids = agent_ids
        self.reset()

    def get_state(self):
        state_dict = {"full_state": self.state}
        for name in self._agent_ids:
            _state_dict = {}
            if name.startswith("high"):
                obs = self.observation_dict_user_high
            elif name.startswith("mid"):
                obs = self.observation_dict_user_mid
            elif name.startswith("low"):
                obs = self.observation_dict_user_low
            else:
                raise ValueError

            for key in obs:
                if key == 'menu' and name.startswith("low"):
                    slot = int(name.split('_')[-1])
                    _state_dict[key] = self.state[key][slot * 2:(slot + 1) * 2]
                else:
                    _state_dict[key] = self.state[key]
            state_dict[name] = _state_dict
        return state_dict

    def get_reward(self, new_state, old_state):
        rewards = dict()

        if self.mid is not None:
            mid = self._get_mid_level_reward(new_state, old_state)
            rewards["mid_level_agent"] = mid

        high_level = self._get_high_level_reward(new_state, old_state)
        low_level = self._get_low_level_reward(new_state, old_state)
        self.add['low'] = low_level
        self.add['low_level'] = 1
        rewards["low_level_agent"] = low_level
        rewards["high_level_agent"] = high_level
        return rewards

    def _get_high_level_reward(self, new_state, old_state):
        if self.done:
            done_reward = 1
        else:
            done_reward = 0
        old_differences = np.sum(old_state['selection'] != old_state['goal'])
        new_differences = np.sum(new_state['selection'] != new_state['goal'])
        high = old_differences - new_differences
        reward = high + done_reward
        self.add['done'] = done_reward
        self.add['high'] = reward
        return reward

    def _get_mid_level_reward(self, new_state, old_state):
        item = np.where(new_state['item_target'] == 1)[0][0]
        slot = np.where(new_state['slot_target'] == 1)[0][0]
        reward = self.item_to_slot[item, slot] - 1
        self.add['mid'] = reward
        return reward

    def _get_low_level_reward(self, new_state, old_state):
        hit = -1
        target = np.where(new_state['slot_target'] == 1)[0][0]
        if check_within_element(new_state['position'], self.ui[target]):
            hit = 0

        target_center = np.asarray(
            [self.ui[target]['x'] + self.ui[target]['w'] / 2, self.ui[target]['y'] + self.ui[target]['h'] / 2])

        distance = calc_distance(
            new_state['position'],
            target_center,
            'l2',
        )
        max_distance = calc_distance(
            np.zeros(2),
            np.ones(2),
            'l2',
        )

        distance_penalty = clip_normalize_reward(
            distance, r_min=0, r_max=max_distance
        )

        time_penalty = 0.
        self.add['distance'] = distance_penalty
        self.add['time_penalty'] = time_penalty
        self.add['hit'] = hit
        reward = hit - 4 * distance_penalty - time_penalty
        return reward

    def step_high_level(self, action):
        if self.mid is not None:
            item_target = np.zeros(self.n_items, dtype=np.int8)
            item_target[action] = 1
            self.state['item_target'] = item_target
        else:
            slot_target = np.zeros(self.n_targets, dtype=np.int8)
            slot_target[action] = 1
            self.state['slot_target'] = slot_target
        return self.get_state(), {}, self.done, self.add

    def step_mid_level(self, action):
        slot_target = np.zeros(self.n_items, dtype=np.int8)
        slot_target[action] = 1
        self.state['slot_target'] = slot_target
        return self.get_state(), {}, self.done, self.add

    def step_low_level(self, action):
        _action = np.asarray(action, dtype=np.float64)
        self.old_state = copy.deepcopy(self.state)
        self.state['position'] = _action
        for slot in self.ui:
            if check_within_element(self.state['position'], self.ui[slot]):
                self.state['selection'][self.ui[slot]['item']] = 1 - self.state['selection'][self.ui[slot]['item']]
                break

        if np.array_equal(self.state['selection'], self.state['goal']):
            self.done = True

        self.state["diff"] = (self.state["selection"] != self.state["goal"]).astype(np.int8)
        return self.get_state(), self.get_reward(self.state, self.old_state), self.done, self.add

    def reset(self):
        if self.random:
            self.generate_random_ui()

        self.state = dict()
        self.state["position"] = self.full_dict['position'].sample()
        self.state["selection"] = self.full_dict['selection'].sample()
        self.state["goal"] = generate_goal(copy.deepcopy(self.state["selection"]), n=self.n_goal_diff)
        self.state["difference"] = (self.state["selection"] != self.state["goal"]).astype(np.int8)

        item_to_slot = generate_item_to_slot(self.n_items, self.n_targets)
        self.item_to_slot = item_to_slot
        for slot in self.ui:
            item = np.where(item_to_slot[:, slot] == 1)[0][0]
            self.ui[slot]['item'] = item

        self.state["item_to_slot"] = item_to_slot.flatten()
        self.state["menu"] = self.get_menu_location()
        self.state["item_target"] = self.full_dict['item_target'].sample()
        self.state["slot_target"] = self.full_dict['slot_target'].sample()
        self.done = False
        self.add['low_level'] = 0
        return self.get_state()

    def render(self, mode="human", close=False):
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
        font = pygame.font.Font('freesansbold.ttf', 16)
        x, y = self.state['position'][0] * self.screen_width, self.state['position'][1] * self.screen_height
        gfxdraw.filled_circle(surf, int(x), int(y), 20, (255, 0, 0))

        for element in self.ui:
            rect = pygame.Rect(self.ui[element]['x'] * self.screen_width,
                               self.ui[element]['y'] * self.screen_height,
                               self.ui[element]['w'] * self.screen_width,
                               self.ui[element]['h'] * self.screen_height)
            if self.state['selection'][self.ui[element]['item']] == 1:
                gfxdraw.box(surf, rect, (200, 200, 200))
            else:
                gfxdraw.box(surf, rect, (100, 100, 100))
            if self.state['goal'][self.ui[element]['item']] == 1:
                rect = pygame.Rect(self.ui[element]['x'] * self.screen_width - 5,
                                   self.ui[element]['y'] * self.screen_height - 5,
                                   self.ui[element]['w'] * self.screen_width + 10,
                                   self.ui[element]['h'] * self.screen_height + 10)
                gfxdraw.rectangle(surf, rect, (200, 0, 0))

        self.screen.blit(surf, (0, 0))
        for element in self.ui:
            t = f"S:{element} I:{self.ui[element]['item']}"
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

    def screenshot(self, suffix="image"):
        import pygame
        images = glob.glob(f"{self.current_folder}/*.jpg")
        n_images = str(len(images)).zfill(5)
        image_name = f"{n_images}.jpg"
        pygame.image.save(self.screen, f"{self.current_folder}/{image_name}")

    def update_folder(self, folder):
        timestamp = int(time.time() * 1e3)
        if not os.path.exists(f"{folder}/rollouts/"):
            os.makedirs(f"{folder}/rollouts/")
        directory = f"{folder}/rollouts/{timestamp}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.current_folder = directory


if __name__ == '__main__':
    env_config = {
        'multi': True,
        'mid': "single",
        'random': False,
        'n_targets': 5,
        'n_items': 7,
        'n_diff': 1
    }
    env = Environment(env_config, [])

    for i in range(100):
        if env.done:
            print(f"Task {i} found")
            env.reset()
        env.render()
        goal = env.state['goal']
        selection = env.state['selection']
        difference = (goal - selection) ** 2
        print(difference)
        item = np.where(difference == 1)[0][0]
        env.step_high_level(item)
        slot = np.where(env.item_to_slot[item, :] == 1)[0][0]
        env.step_mid_level(slot)
        location = np.asarray([env.ui[slot]['x'] + env.ui[slot]['w'] / 2, env.ui[slot]['y'] + env.ui[slot]['h'] / 2])
        _, r, _, _ = env.step_low_level(location)
        goal = env.state['goal']
        selection = env.state['selection']
        difference = (goal - selection) ** 2
        time.sleep(0.5)
    time.sleep(10)
    env.close()
