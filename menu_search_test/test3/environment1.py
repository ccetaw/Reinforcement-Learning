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



"""
Setting:
    Stochastic target
"""


class Environment(gym.Env, ABC, Interface):

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 2}
    reward_range = (-float("inf"), float("inf"))
    
    def __init__(self, envconfig):
        super().__init__(envconfig)

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
        )
        obs_space_dict = {
            'cursor_position': spaces.Box(low=0, high=1, shape=(2,)),
            'menu': spaces.Box(low=np.array([0.0, 0.0] * self.n_slots), high=np.array([1.0, 1.0] * self.n_slots), dtype=np.float64),
            'target_slot': spaces.MultiBinary(self.n_slots),
        } 
        self.observation_space = spaces.Dict(obs_space_dict)

        self.done = False
        self.counter = 0
        self.state = {}  # State = Observation in MDP
        self.add = {} # everything else

        self.screen = None
        self.clock = None
        self.isopen = True

    def get_reward(self):
        hit = -1
        target = np.where(self.state['target_slot'] == 1)[0][0]
        if check_within_element(self.state['cursor_position'], self.ui[target]):
            hit = 0

        target_center = np.asarray(
            [self.ui[target]['x'] + self.ui[target]['w'] / 2, self.ui[target]['y'] + self.ui[target]['h'] / 2])

        distance = calc_distance(
            self.state['cursor_position'],
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
        reward = hit - 4 * distance_penalty - time_penalty
        return reward

    def step(self, action):
        _action = np.asarray(action, dtype=np.float64)
        stochastic_action, _ = compute_stochastic_position(_action, self.state['cursor_position'], sigma=0.01)
        self.state['cursor_position'] = stochastic_action
        target = np.where(self.state['target_slot'] == 1)[0][0]
        reward = self.get_reward()
        if check_within_element(self.state["cursor_position"], self.ui[target]):
            self.done = True
        elif self.counter > 10:
            self.done = True
            reward = reward - 5
        self.counter += 1
        self.add['deterministic_action'] = _action
        return self.state, reward, self.done, {}

    def reset(self):
        slot = np.zeros(self.n_slots)
        idx = np.random.randint(self.n_slots)
        slot[idx] =  1
        self.state['target_slot'] = slot
        self.state['cursor_position'] = np.random.random_sample(size=(2,))
        if self.random:
            self.generate_random_ui()
        else:
            self.generate_static_ui()
        self.state['menu'] = self.get_menu_location()
        self.add['deterministic_action'] = np.array([0.0,0.0])

        self.done = False
        self.counter = 0
        return self.state

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
        font = pygame.font.Font('freesansbold.ttf', 16)
        x, y = self.state['cursor_position'][0] * self.screen_width, self.state['cursor_position'][1] * self.screen_height
        slot = np.argmax(self.state["target_slot"])
        if check_within_element(self.state["cursor_position"], self.ui[slot]):
            gfxdraw.filled_circle(surf, int(x), int(y), 20, (0, 0, 255))
        else:
            gfxdraw.filled_circle(surf, int(x), int(y), 20, (255, 0, 0))

        if self.add['deterministic_action'] is not None:
            x, y = self.add['deterministic_action'][0] * self.screen_width, self.add['deterministic_action'][1] * self.screen_height
            gfxdraw.circle(surf, int(x), int(y), 20, (0, 150, 150))

        for element in self.ui:
            rect = pygame.Rect(self.ui[element]['x'] * self.screen_width,
                               self.ui[element]['y'] * self.screen_height,
                               self.ui[element]['w'] * self.screen_width,
                               self.ui[element]['h'] * self.screen_height)
            if element == slot:
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
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    args = parser.parse_args()

    env_config = {
        'random':True,
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
            "training_iteration": 300,
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
            if  result["episode_reward_mean"] >= stop["episode_reward_mean"]:
                break

        trainer.save("test3_1")

        env = Environment(env_config)
        for i in range(10):
            env.reset()
            while not env.done:
                action = trainer.compute_single_action(env.state)
                env.step(action)
                env.render()
        env.close()

        