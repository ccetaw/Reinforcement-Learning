from abc import ABC
import gym
import numpy as np
from utils.env_utils import (
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
from envs.lightswitches.interface import Interface
import copy
import glob
import os
import time


class Environment(gym.Env, ABC, Interface):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_config, agent_ids):
        super().__init__(env_config)
        self.add = {}  # whatever additional information there is
        self.state = None
        self.done = False
        self.lower_level_step_counter = 0
        self.ep_counter = 0
        self.old_state, self.actions, self.new_state, self.rewards = [], [], [], []
        self.screen = None
        self.clock = None
        self.isopen = True
        self.random = env_config['random']
        if not self.random:
            self.generate_static_ui()
        self._agent_ids = agent_ids
        self.observation_space_high = ['position', 'item_to_slot', 'selection', 'goal', 'difference']
        self.observation_space_mid = ['item_target', 'item_to_slot', 'position']
        self.observation_space_low = ['position']

        self.task = 1
        self.since_last_update = 0
        self.lower_level_step_counter = 0
        if self.low == 'coordinate':
            self.observation_space_low.append('target_coordinate')
        else:
            self.observation_space_low.extend(['slot_target', 'menu'])
        self.reset()

    def get_state(self):
        state_dict = {"full_state": self.state}
        for name in self._agent_ids:
            _state_dict = {}
            if name.startswith("high"):
                obs = self.observation_space_high
            elif name.startswith("mid"):
                obs = self.observation_space_mid
            elif name.startswith("low"):
                obs = self.observation_space_low
            else:
                raise ValueError
            for key in obs:
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
        done_reward = 1 if self.done else 0
        goal = new_state["goal"]
        selection = copy.deepcopy(new_state["selection"])
        difference_penalty = np.sum((goal - selection) ** 2)
        difference_penalty = -1 * clip_normalize_reward(difference_penalty, 0, self.n_items)
        reward = (difference_penalty
                  - clip_normalize_reward((self.mt + self.decision_time), 0, 3)
                  + done_reward)
        self.add['done'] = done_reward
        self.add['high'] = reward
        return reward

    def _get_mid_level_reward(self, new_state, old_state):
        item = np.where(new_state['item_target'] == 1)[0][0]
        slot = np.where(new_state['slot_target'] == 1)[0][0]
        reward = self.item_to_slot[item, slot]
        self.add['mid'] = reward
        return reward

    def _get_low_level_reward(self, new_state, old_state):
        target = np.where(new_state['slot_target'] == 1)[0][0]
        hit_reward = 1 if check_within_element(new_state['position'], self.ui[target]) else 0
        target_center = np.asarray(
            [self.ui[target]['x'] + self.ui[target]['w'] / 2, self.ui[target]['y'] + self.ui[target]['h'] / 2])

        distance = calc_distance(
            new_state['position'],
            target_center,
            'l2',
        ) ** 0.5
        max_distance = calc_distance(
            np.zeros(2),
            np.ones(2),
            'l2',
        ) ** 0.5
        distance_penalty = clip_normalize_reward(
            distance, r_min=0, r_max=max_distance
        )
        time_penalty = - clip_normalize_reward(self.mt, 0, 3)
        self.add['distance'] = distance
        self.add['time_penalty'] = time_penalty
        self.add['mt'] = self.mt
        self.add['hit'] = hit_reward
        weight = 0.95
        reward = (weight * hit_reward
                  + (1 - weight) * time_penalty
                  - 8. * distance_penalty)
        return reward

    def step_high_level(self, action):
        if self.mid is not None:
            item_target = np.zeros(self.n_items, dtype=np.int8)
            item_target[action] = 1
            self.state['item_target'] = item_target
        else:
            slot_target = np.zeros(self.n_slots, dtype=np.int8)
            slot_target[action] = 1
            self.state['slot_target'] = slot_target
        return self.get_state(), {}, self.done, self.add

    def step_mid_level(self, action):
        if self.mid == 'heuristic':
            item = np.where(self.state['item_target'] == 1)[0][0]
            action = np.where(self.item_to_slot[item, :] == 1)[0][0]

        target_center = np.asarray(
            [self.ui[action]['x'] + self.ui[action]['w'] / 2, self.ui[action]['y'] + self.ui[action]['h'] / 2])
        self.state['target_coordinate'] = target_center
        slot_target = np.zeros(self.n_items, dtype=np.int8)
        slot_target[action] = 1
        self.state['slot_target'] = slot_target
        return self.get_state(), {}, self.done, self.add

    def step_low_level(self, action):
        self.old_state = copy.deepcopy(self.state)
        self.lower_level_step_counter += 1
        if self.low == 'heuristic':
            slot = np.where(self.state['slot_target'] == 1)[0][0]
            current_pos = action[:2]
            width, distance = compute_width_distance_fast(current_pos, self.ui[slot])
            sigma = width / 4
            self.mt = who_mt(distance, sigma)
            target_center = np.asarray(
                [self.ui[slot]['x'] + self.ui[slot]['w'] / 2, self.ui[slot]['y'] + self.ui[slot]['h'] / 2])
            self.state['position'] = np.clip(np.random.normal(target_center, sigma), 0.0, 1.0)

        else:
            action = np.asarray(action, dtype=np.float64)
            if self.relative:
                new_desired = self.state['position'] + action[:2]
            else:
                new_desired = action[:2]

            old_pos = self.state["position"].copy()
            sigma = scale(action[2], 0.001, 0.1)
            sampled_pos, self.mt = compute_stochastic_position(new_desired, old_pos, sigma)
            self.state['position'] = sampled_pos

        for slot in self.ui:
            if check_within_element(self.state['position'], self.ui[slot]):
                self.state['selection'][self.ui[slot]['item']] = 1 - self.state['selection'][self.ui[slot]['item']]
                break

        if np.array_equal(self.state['selection'], self.state['goal']):
            self.done = True
            self.add['steps per diff'] = self.lower_level_step_counter / self.add['n_diff']
        self.state["difference"] = (self.state["selection"] - self.state["goal"]) ** 2

        return self.get_state(), self.get_reward(self.state, self.old_state), self.done, self.add

    def reset(self):
        if self.random:
            self.ui = self.generate_random_ui()
        self.lower_level_step_counter = 0

        self.state = dict()
        self.state['target_coordinate'] = np.random.uniform(0, 1, size=2).astype(np.float64)
        self.state["position"] = np.random.uniform(0, 1, size=2).astype(np.float64)
        self.state["selection"] = np.random.choice([0, 1], size=self.n_items)
        n = self.diff if self.diff != 'curriculum' else self.task
        self.state["goal"], self.add['n_diff'] = generate_goal(copy.deepcopy(self.state["selection"]), n=n, curriculum=self.diff=='curriculum')
        self.state["difference"] = (self.state["selection"] - self.state["goal"]) ** 2

        item_to_slot = generate_item_to_slot(self.n_items, self.n_slots)
        self.item_to_slot = item_to_slot
        for slot in self.ui:
            item = np.where(item_to_slot[:, slot] == 1)[0][0]
            self.ui[slot]['item'] = item

        self.state["item_to_slot"] = item_to_slot.flatten()
        self.state["menu"] = self.get_menu_location()
        item_target = np.zeros(self.n_items)
        item = np.random.randint(self.n_items)
        item_target[item] = 1
        slot_target = np.zeros(self.n_slots)
        slot = np.random.randint(self.n_slots)
        slot_target[slot] = 1
        self.state["item_target"] = item_target
        self.state["slot_target"] = slot_target
        self.done = False
        self.add['low_level'] = 0
        self.add['task'] = self.task
        self.add['since_last_update'] = self.since_last_update
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
        'low': True,
        'random': True,
        'mid': True,
        'n_targets': 10,
        'n_items': 10,
        'n_diff': 'random'
    }
    env = Environment(env_config, [])
    postion = np.random.uniform(0, 1, size=(2,))
    print("starting")
    s = time.time()
    for i in range(100):
        compute_width_distance(postion, env.ui[0])
    e = time.time()
    print(e - s)
    s = time.time()
    for i in range(100):
        compute_width_distance_fast(postion, env.ui[0])
    e = time.time()
    print(e - s)
