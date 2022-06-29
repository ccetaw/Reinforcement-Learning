from ray.tune.trial import Trial
from ray.tune import ExperimentAnalysis
import glob


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"{agent_id}_policy"


def trial_name_string(trial: Trial) -> str:
    env_config = trial.config["env_config"]
    keys = list(env_config.keys())
    trial_name = f"{trial.trial_id}"
    for key in keys:
        trial_name += f"-{key}_{env_config[key]}"
    return trial_name


def get_checkpoint(data: str or ExperimentAnalysis) -> (dict, dict):
    if isinstance(data, str):
        analysis = ExperimentAnalysis(data)
    else:
        analysis = data

    trials = analysis.trials
    print(trials)
    paths = {}
    configs = {}
    for trial in trials:
        _trial = str(trial)
        root = f"{data }/{_trial}"
        checkpoints = glob.glob(f"{root}_*/checkpoint_*", recursive=True)
        ids = [x.split("_")[-1] for x in checkpoints]
        paths[_trial] = []
        configs[_trial] = trial.config
        for c, id in zip(checkpoints, ids):
            checkpoint_id = id.lstrip("0")
            path = c + f"/checkpoint-{str(checkpoint_id)}"
            paths[_trial].append(path)
    return paths, configs
