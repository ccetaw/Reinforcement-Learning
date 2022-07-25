from gym import spaces
import numpy as np
from interface import Interface
from utils import (
    calc_distance,
    compute_stochastic_position
    )
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from datetime import datetime
from utils import policy_mapping_fn

class MultiAgentUI(MultiAgentEnv, Interface):

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 1}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, envconfig, init=True):
        if init:
            super(MultiAgentEnv, self).__init__(envconfig)
            self.save_ui = envconfig['save_ui']
        else:
            uiconfig = {
                'random': False,
                'n_buttons': 7
            }
            super(MultiAgentEnv, self).__init__(uiconfig)
            self.save_ui = False

        self._agent_ids = ['user_high', 'user_low']
        self.action_space = {
            'user_high': spaces.Discrete(self.n_buttons),
            'user_low': spaces.Box(low=0, high=1, shape=(2,), dtype=float)
        }
        obs_user_high_dict = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons)
        }
        obs_user_low_dict = {
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'target': spaces.Box(low=0., high=1., shape=(2,))
        }
        
        self.observation_space = {
            'user_high': spaces.Dict(obs_user_high_dict),
            'user_low': spaces.Dict(obs_user_low_dict)
        }

        self.state = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons),
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'target': spaces.Box(low=0., high=1., shape=(2,)),
            'menu': self.ui
        }

        self.counter = 0
        self.done = False
        
        if self.save_ui:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = dt_string + f"n_buttons-{self.n_buttons}_" + f"random-{self.random}" + ".json"
            self.save(file_name)

        self.screen = None
        self.clock = None
        self.isopen = True

        self.reset()

    def get_obs(self, agent_id):
        obs = {}
        for key in self.observation_space[agent_id]:
            obs[key] = self.state[key]
        return {agent_id: obs}

    def get_reward(self, agent_id):
        pass

    def reset(self):
        self.state['current_pattern'] = spaces.MultiBinary(self.n_buttons).sample()
        self.state['goal_pattern'] = spaces.MultiBinary(self.n_buttons).sample()
        self.state['cursor_position'] = spaces.Box(low=0., high=1., shape=(2,)).sample()
        self.counter = 0
        self.done = False

        return self.get_obs('user_high')

    def step(self, action_dict):
        agent_active = str(list(action_dict.keys())[0])
        action = list(action_dict.values())[0]
        reward = {}
        if agent_active == 'user_high':
            to_swtich = np.nonzero(self.state['goal_pattern'] - self.state['current_pattern'])[0]
            if action in to_swtich:
                reward['user_high'] = 0
            else:
                reward['user_high'] = -5
            target = self.button_normalized_position(self.ui[action]) + 0.5 * self.button_normalized_size(self.ui[action])
            self.state['target'] = target
            self.counter += 1
            return self.get_obs('user_low'), reward, {"__all__": False}, {}
        elif agent_active == 'user_low':
            self.state['cursor_position'], _ = compute_stochastic_position(action, self.state['cursor_position'], sigma=0.01)
            reward['user_low'] = -calc_distance(self.state['cursor_position'], self.state['target'], 'l2')
            in_button = self.check_within_button(self.state['cursor_position'])
            if in_button is not None:
                self.state['current_pattern'][in_button] = 1 - self.state['current_pattern'][in_button]
                # print(f"Button id {in_button}")

            if np.array_equal(self.state['current_pattern'], self.state['goal_pattern']):
                self.done = True
                done = {'__all__': True}
                reward['user_high'] = -self.counter
            elif self.counter >= self.n_buttons * 2:
                self.done = True
                done = {'__all__': True}
                reward['user_high'] = -self.counter * 2
            else:
                done = {'__all__': False}
            return self.get_obs('user_high'), reward, done, {}


    def render(self, mode='human'):
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

        x, y = self.state['cursor_position'][0] * self.screen_height, self.state['cursor_position'][1] * self.screen_width
        gfxdraw.filled_circle(surf, int(y), int(x), 20, (0, 0, 255))
        x, y = self.state['target'][0] * self.screen_height, self.state['target'][1] * self.screen_width
        gfxdraw.circle(surf, int(y), int(x), 20, (0, 0, 255))
            
        
        for button in self.ui:
            rect = pygame.Rect(button.position[1],
                                    button.position[0],
                                    button.size[1],
                                    button.size[0])
            if button.id in np.nonzero(self.state['current_pattern'])[0]:
                gfxdraw.rectangle(surf, rect, (255, 0, 0))  # Red means on
            else:
                gfxdraw.rectangle(surf, rect, (0, 0, 0))   # Black means off
        self.screen.blit(surf, (0, 0))

        for button in self.ui:
            if button.id in np.nonzero(self.state['goal_pattern'])[0]:
                t = f"Button {button.id} to ON"
            else:
                t = f"Button {button.id} to OFF"
            text = font.render(t, True, (0, 0, 0))
            self.screen.blit(text, (button.position[1], button.position[0]))


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




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    args = parser.parse_args()

    uiconfig = {
        'random': True,
        'n_buttons': 10,
        'save_ui': False
    }
    if args.demo:
        env = MultiAgentUI(uiconfig)

        for _ in range(3):
            env.reset()
            for _ in range(10):
                env.step({'user_high': env.action_space['user_high'].sample()})
                env.step({'user_low': env.state['target']})
                env.render()
        env.close()

    if args.dry_train:
        from ray.rllib.agents import ppo
        from ray.tune.logger import pretty_print
        env = MultiAgentUI(uiconfig)
        policies = {
            "user_high": (
                None,
                env.observation_space['user_high'],
                env.action_space['user_high'],
                {}
            ),
            "user_low":(
                None,
                env.observation_space['user_low'],
                env.action_space['user_low'],
                {}
            )
        }

        config = {
            "num_workers": 3,
            "env": MultiAgentUI,
            "env_config": uiconfig,
            "gamma": 0.9,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "framework": "torch",
            "log_level": "CRITICAL",
            "num_gpus": 0,
            "seed": 31415,
        }

        stop = {
            "training_iteration": 150,
        }

        config = {**ppo.DEFAULT_CONFIG, **config}
        trainer = ppo.PPOTrainer(config=config)

        for _ in range(stop["training_iteration"]):
            result = trainer.train()
            print(pretty_print(result))

        trainer.save("test4_1")

        # env.load("n_buttons-7random-False.json")
        # for i in range(10):
        #     env.reset()
        #     while not env.done:
        #         action = trainer.compute_single_action(env.get_obs('user_high'))
        #         env.step(action)
        #         action = trainer.compute_single_action(env.get_obs('user_low'))
        #         env.step(action)
        #         env.render()
        # env.close()


