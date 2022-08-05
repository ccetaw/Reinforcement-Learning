from env import ButtonPanel
from ray.rllib.agents import ppo
import numpy as np
from tqdm import tqdm



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
env = ButtonPanel(env_config)

best_10_ui = np.ones(10) * 31200000
for _ in tqdm(range(15000), desc="n_buttons=6, mode=all_exclu"):
    env.generate_random_ui()
    sum_time = 0
    for i in range(100):  
        env.reset(after_train=True)
        while not env.done:
            action = agent.compute_single_action(env.get_obs(), unsquash_action=True)
            env.step(action, after_train=True)
            sum_time += env.add['move_time']
    idx = np.nonzero(best_10_ui > sum_time)[0]
    if len(idx) != 0:
        idx = idx[0]
        best_10_ui[idx] = sum_time
        env.save('./best_ui/n_buttons_6-mode_all_exclu_top'+str(idx))
print(best_10_ui)

    

