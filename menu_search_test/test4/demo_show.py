from env3 import MultiAgentUI
from ray.rllib.agents import ppo
from utils import policy_mapping_fn

env_config = {
        'random': False,
        'n_buttons': 6,
        'save_ui': False,
        'mode': 'all_exclu'
    }
env = MultiAgentUI(env_config)

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
            "env_config": env_config,
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
config = {**ppo.DEFAULT_CONFIG, **config}
agent = ppo.PPOTrainer(config=config)
agent.restore('/home/cetaceanw/repo/Reinforcement-Learning/menu_search_test/test4/test4_1/checkpoint_000100/checkpoint-100')


for i in range(10):
    env.reset()
    env.render()
    while not env.done:
        action = agent.compute_single_action(observation=list(env.get_obs('user_high').values())[0], policy_id='user_high', unsquash_action=True)
        env.step({'user_high': action})
        env.step({'user_low': env.state['target']})
        env.render()
env.close()