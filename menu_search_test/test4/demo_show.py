from env import MultiAgentUI
from ray.rllib.agents import ppo
from utils import policy_mapping_fn

env_config = {
        'random': False,
        'n_buttons': 7,
        'save_ui': False
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
agent.restore('test4_1/checkpoint_000150/checkpoint-150')


for i in range(10):
    env.reset()
    while not env.done:
        action = agent.get_policy('user_high').compute_single_action(obs= env.get_obs('user_high'))
        env.step(action)
        # action = agent.compute_single_action(env.get_obs('user_low'), policy_id=policy_mapping_fn('user_low'))
        action = agent.get_policy('user_low').compute_single_action(obs= env.get_obs('user_low'))
        env.step(action)
        env.render()
env.close()