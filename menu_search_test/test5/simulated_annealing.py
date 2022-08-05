from simanneal import Annealer
from env import ButtonPanel
import numpy as np
from ray.rllib.agents import ppo

class BestUI(Annealer, ButtonPanel):
    def __init__(self, config):
        super(Annealer, self).__init__(config)
        super().__init__(initial_state=self.button_parameters())
        self.agent = config['agent']

    def move(self):
        flag = True
        state = self.button_parameters().copy()
        to_state = self.button_parameters().copy()
        while flag:
            for i in range(self.n_buttons):
                to_move_0 = state[i]['position'][0] + np.random.randint(low=-1, high=2) * self.grid_size
                to_state[i]['position'][0] = to_move_0 if to_move_0 > self.margin_size and to_move_0 < self.screen_height - self.margin_size else state[i]['position'][0]
                to_move_1 = state[i]['position'][1] + np.random.randint(low=-1, high=2) * self.grid_size
                to_state[i]['position'][1] = to_move_1 if to_move_1 > self.margin_size and to_move_1 < self.screen_width - self.margin_size else state[i]['position'][1]
                to_height = state[i]['height'] + np.random.randint(low=-1, high=2) 
                to_state[i]['height'] = to_height if to_height>=0 and to_height<=2 else state[i]['height']
                to_width = state[i]['width'] + np.random.randint(low=-1, high=2) 
                to_state[i]['width'] = to_width if to_width>=0 and to_width<=2 else state[i]['width']
            flag = not self.set_from_button_parameters(to_state)
        self.state = self.button_parameters()


    def energy(self):
        sum_time = 0
        for _ in range(300):  
            self.reset(after_train=True)
            while not self.done:
                action = self.agent.compute_single_action(self.get_obs(), unsquash_action=True)
                self.step(action, after_train=True)
                sum_time += self.add['move_time']
        return sum_time

env_config = {
    'random': True,
    'n_buttons': 6,
    'mode': 'all_exclu'
}

config = {
    "num_workers": 3,
    "env": ButtonPanel,
    "env_config": env_config,
    "gamma": 0.9,
    "framework": "torch",
    "log_level": "CRITICAL",
    "num_gpus": 0,
    "seed": 31415,
}

config = {**ppo.DEFAULT_CONFIG, **config}
agent = ppo.PPOTrainer(config=config)
agent.restore('./trained_user/PPOTrainer_2022-08-01_20-04-46/random_True-n_buttons_6-save_ui_True-mode_all_exclu/checkpoint_000200/checkpoint-200')

optim_config = {
    **env_config,
    'agent': agent
}

optimizer = BestUI(optim_config)
optimizer.Tmax = 9000
optimizer.Tmin = 0.5
optimizer.steps = 10000

optimizer.anneal()
optimizer.save('./sa_1')


    