from gym import spaces
import numpy as np
from interface import Interface
from utils import (
    compute_stochastic_position,
    minjerk_trajectory,
    compute_width_distance_fast,
    jerk_of_minjerk_trajectory
    )
import gym

class ButtonPanel(gym.Env, Interface):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 60}
    reward_range = (-float("inf"), float("inf"))
    user_pattern_5 = {
        'normal':       np.array([[1,0,0,0,0], [1,1,0,0,0], [0,0,1,0,0], [0,0,0,1,1], [0,0,0,0,1]]),
        'central':      np.array([[1,1,1,1,1], [0,0,0,0,0], [0,1,0,0,0], [0,0,1,1,0], [0,0,0,0,1]]),
        'mutual_exclu': np.array([[1,0,0,0,0], [0,1,0,0,0], [1,0,0,0,1], [1,0,0,1,0], [0,1,0,0,1]]),
        'all_exclu':    np.array([[1,0,0,0,0], [0,1,1,1,1], [0,1,0,0,0], [0,0,0,1,1], [0,0,1,0,0]])
    }
    user_pattern_6 = {
        'normal':       np.array([[0,0,0,0,0,0], [1,1,0,0,0,0], [1,0,1,0,0,0], [0,1,0,1,0,0], [0,0,0,1,0,1], [0,0,0,0,1,0]]),
        'central':      np.array([[1,1,1,1,1,1], [0,0,0,0,0,0], [0,1,1,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,1], [0,1,0,0,0,1]]),
        'mutual_exclu': np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [1,0,0,1,0,0], [0,1,0,0,1,0], [0,1,0,0,0,1]]),
        'all_exclu':    np.array([[1,0,0,0,0,0], [0,1,1,1,1,1], [0,1,1,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,1], [0,1,0,0,0,1]])
    }
    user_pattern_7 = {
        'normal':       np.array([[1,0,0,0,0,0,0], [1,0,0,1,0,0,0], [0,1,1,0,0,0,0], [0,0,0,0,1,0,1], [0,0,0,1,0,1,1], [0,0,0,0,1,1,0], [0,0,0,0,0,0,1]]),
        'central':      np.array([[1,1,1,1,1,1,1], [0,0,0,0,0,0,0], [0,1,0,0,0,1,0], [0,1,0,1,0,0,0], [0,0,1,0,1,0,0], [0,0,0,0,1,0,1], [0,0,0,0,0,1,0]]),
        'mutual_exclu': np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [1,0,0,1,0,0,0], [0,1,0,0,0,1,1], [0,0,1,0,1,0,0], [0,0,0,0,1,0,1]]),
        'all_exclu':    np.array([[1,0,0,0,0,0,0], [0,1,1,1,1,1,1], [0,1,0,0,0,0,0], [0,0,1,1,1,0,0], [0,1,1,0,0,0,0], [0,0,0,1,0,0,1], [0,0,0,0,0,1,1]])
    }
    user_pattern = {}
    user_pattern[5] = user_pattern_5
    user_pattern[6] = user_pattern_6
    user_pattern[7] = user_pattern_7
    probabilities = {}
    probabilities[5] = [33, 15, 20, 17, 15]
    probabilities[6] = [33, 25, 15, 10, 10, 7]
    probabilities[7] = [33, 25, 15, 7, 5, 5, 10]
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.action_space = spaces.Discrete(self.n_buttons)
        obs_dict = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons)
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.obs_full = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons),
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'menu': self.ui
        }

        self.counter = 0
        self.done = False
        self.add = {}
        self.dt = 0.01

        self.screen = None
        self.clock = None
        self.isopen = True

    def get_obs(self):
        obs = {}
        for key in self.observation_space:
            obs[key] = self.obs_full[key]
        return obs

    def step(self, action, after_train=False):
        if after_train:
            self.add['previous_pattern'] = self.obs_full['current_pattern'].copy()
            self.obs_full['target'] = self.button_normalized_position(self.ui[action]) + 0.5 * self.button_normalized_size(self.ui[action])
            sigma, _ = compute_width_distance_fast(self.obs_full['cursor_position'], self, action)
            self.add['move_from'] = self.obs_full['cursor_position']
            self.add['move_to'], self.add['move_time'] = compute_stochastic_position(self.obs_full['target'], self.obs_full['cursor_position'], sigma/4)
            self.add['jerk'] = jerk_of_minjerk_trajectory(self.add['move_time'], self.add['move_from'], self.add['move_to'])
            in_button = self.check_within_button(self.add['move_to'])
            self.press(in_button, self.obs_full['current_pattern'])
        else:
            self.press(action, self.obs_full['current_pattern'])
        reward = -1
        self.counter += 1
        if np.array_equal(self.obs_full['current_pattern'], self.obs_full['goal_pattern']):
            self.done = True
            reward += self.n_buttons * 2
        elif self.counter >= self.n_buttons * 2:
            self.done = True
            reward -= self.n_buttons * 2
        
        return self.get_obs(), reward, self.done, {}

    def reset(self, after_train=False, start_pattern=None, goal_pattern=None):
        if after_train:
            if start_pattern is not None and goal_pattern is not None:
                self.obs_full['current_pattern'] = start_pattern
                self.obs_full['goal_pattern'] = goal_pattern
            else:
                cum_sum = np.cumsum(self.probabilities[self.n_buttons])
                idx1 = np.random.randint(low=1, high=101)
                to_choose1 = np.nonzero( (cum_sum - idx1) >= 0)[0][0]
                self.obs_full['current_pattern'] = self.user_pattern[self.n_buttons][self.mode][to_choose1].copy()
                idx2 = np.random.randint(low=1, high=101)
                to_choose2 = np.nonzero( (cum_sum - idx2) >= 0)[0][0]
                while to_choose2 == to_choose1:
                    idx2 = np.random.randint(low=1, high=101)
                    to_choose2 = np.nonzero( (cum_sum - idx2) >= 0)[0][0]
                self.obs_full['goal_pattern'] = self.user_pattern[self.n_buttons][self.mode][to_choose2].copy()
        else:
            self.obs_full['current_pattern'] = self.sample_possible_pattern()
            self.obs_full['goal_pattern'] = self.sample_possible_pattern()

        self.obs_full['cursor_position'] = np.array([0.,0.])
        self.counter = 0
        self.done = False

        return self.get_obs()
    
    def render(self):
        import pygame
        from pygame import gfxdraw

        if self.obs_full is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        time = 0
        arrived = False
        while not arrived:
            if time + self.dt < self.add['move_time']:
                time += self.dt
            else:
                time = self.add['move_time']
                arrived = True
            
            self.obs_full['cursor_position'] = minjerk_trajectory(time, self.add['move_time'], self.add['move_from'] ,self.add['move_to'])
            surf = pygame.Surface((self.screen_width, self.screen_height))
            surf.fill((255, 255, 255))
            font = pygame.font.Font('freesansbold.ttf', 16)
            x, y = self.obs_full['cursor_position'][0] * self.screen_height, self.obs_full['cursor_position'][1] * self.screen_width
            gfxdraw.filled_circle(surf, int(y), int(x), 20, (0, 0, 255))
            x, y = self.obs_full['target'][0] * self.screen_height, self.obs_full['target'][1] * self.screen_width
            gfxdraw.circle(surf, int(y), int(x), 20, (0, 0, 255))
                
            if arrived:
                for button in self.ui:
                    rect = pygame.Rect(button.position[1],
                                            button.position[0],
                                            button.size[1],
                                            button.size[0])
                    if button.id in np.nonzero(self.obs_full['current_pattern'])[0]:
                        gfxdraw.rectangle(surf, rect, (255, 0, 0))  # Red means on
                    else:
                        gfxdraw.rectangle(surf, rect, (0, 0, 0))   # Black means off
            else:
                for button in self.ui:
                    rect = pygame.Rect(button.position[1],
                                            button.position[0],
                                            button.size[1],
                                            button.size[0])
                    if button.id in np.nonzero(self.add['previous_pattern'])[0]:
                        gfxdraw.rectangle(surf, rect, (255, 0, 0))  # Red means on
                    else:
                        gfxdraw.rectangle(surf, rect, (0, 0, 0))   # Black means off
            self.screen.blit(surf, (0, 0))

            for button in self.ui:
                t = f"Button {button.id}"
                if button.id in np.nonzero(self.obs_full['goal_pattern'])[0]:
                    text = font.render(t, True, (255, 0, 0))
                else:
                    text = font.render(t, True, (0, 0, 0))
                self.screen.blit(text, (button.position[1], button.position[0]))

            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    args = parser.parse_args()

    env_config = {
        'random': False,
        'n_buttons': 6,
        'mode': 'all_exclu'
    }
    if args.demo:
        env = ButtonPanel(env_config)

        for _ in range(3):
            env.reset(after_train=True)
            for _ in range(10):
                env.step(env.action_space.sample(), after_train=True)
                env.render()
        env.close()

    if args.dry_train:
        from ray.rllib.agents import ppo
        from ray.tune.logger import pretty_print
        config = {
            "env": ButtonPanel,  # or "corridor" if registered above
            "env_config": env_config,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 3,  # parallelism
            "log_level": "CRITICAL",
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
        trainer = ppo.PPOTrainer(config=ppo_config)
        # run manual training loop and print results after each iteration
        for _ in range(stop["training_iteration"]):
            result = trainer.train()
            print(pretty_print(result))

        trainer.save("test5_1")

        env = ButtonPanel(env_config)
        for i in range(10):
            env.reset()
            while not env.done:
                action = trainer.compute_single_action(env.state)
                env.step(action)
                env.render()
        env.close()