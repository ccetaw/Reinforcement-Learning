from simanneal import Annealer
from env import ButtonPanel
import numpy as np
from ray.rllib.agents import ppo
import matplotlib.pyplot as plt
from utils import calc_distance

class BestUI(Annealer, ButtonPanel):
    def __init__(self, config):
        super(Annealer, self).__init__(config)
        super().__init__(initial_state=self.button_parameters())
        self.agent = config['agent']
        self.history = []

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
                to_state[i]['width'] = to_state[i]['height']
            flag = not self.set_from_button_parameters(to_state)
        self.state = self.button_parameters()


    def energy(self):
        sum_dist = 0
        for _ in range(500):  
            self.reset(after_train=True)
            while not self.done:
                action = self.agent.compute_single_action(self.get_obs(), unsquash_action=True)
                self.step(action, after_train=True)
                sum_dist += calc_distance(self.add['move_to'], self.add['move_from'], method='l2')
        self.history.append(sum_dist)
        return sum_dist

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
optimizer.generate_static_ui()
auto_schedule = optimizer.auto(minutes=1)
optimizer.set_schedule(auto_schedule)


optimizer.anneal()
optimizer.save('./sa_2')

x = np.array(range(len(optimizer.history)))
y = np.array(optimizer.history)

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.show()


    