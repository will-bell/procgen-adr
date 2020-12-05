import copy
import pathlib
from typing import List, Tuple

import numpy as np
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from procgen import ProcgenEnv
from procgen.domains import BossfightHardConfig, BossfightEasyConfig

from baselines_adr.adr_runner import EnvironmentParameter, DEFAULT_DOMAIN_CONFIGS
from baselines_adr.adr_runner import safemean


class TestRunner:
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, model, config_dir: pathlib.Path, n_trajectories: int, tunable_params: List[EnvironmentParameter]):
        self._model = model
        self._n_trajectories = n_trajectories

        # Initialize the environment
        easy_config_path = config_dir / 'test_easy_config.json'
        easy_config = copy.copy(BossfightEasyConfig)
        easy_config.to_json(easy_config_path)
        easy_env = ProcgenEnv(num_envs=1, env_name=str(easy_config.game), domain_config_path=str(easy_config_path))
        easy_env = VecExtractDictObs(easy_env, "rgb")
        easy_env = VecMonitor(venv=easy_env, filename=None, keep_buf=100)
        self.easy_env = VecNormalize(venv=easy_env, ob=False)

        hard_config_path = config_dir / 'test_hard_config.json'
        hard_config = copy.copy(BossfightHardConfig)
        hard_config.to_json(hard_config_path)
        hard_env = ProcgenEnv(num_envs=1, env_name=str(hard_config.game), domain_config_path=str(hard_config_path))
        hard_env = VecExtractDictObs(hard_env, "rgb")
        hard_env = VecMonitor(venv=hard_env, filename=None, keep_buf=100)
        self.hard_env = VecNormalize(venv=hard_env, ob=False)

        # Make a default config for bossfight...
        test_domain_config_path = config_dir / 'test_full_config.json'
        test_domain_config = DEFAULT_DOMAIN_CONFIGS['dc_bossfight']
        test_domain_config.to_json(test_domain_config_path)

        params = {}
        for param in tunable_params:
            params['min_' + param.name] = param.clip_lower_bound
            params['max_' + param.name] = param.clip_upper_bound
        test_domain_config.update_parameters(params, cache=False)

        full_env = ProcgenEnv(num_envs=1, env_name=str(test_domain_config.game), domain_config_path=str(test_domain_config_path))
        full_env = VecExtractDictObs(full_env, "rgb")
        full_env = VecMonitor(venv=full_env, filename=None, keep_buf=100)
        self.full_env = VecNormalize(venv=full_env, ob=False)

    def run(self) -> Tuple[float, float, float]:
        easy_rewards = self.run_env(self.easy_env)
        easy_mean_rew = safemean(easy_rewards)
        
        hard_rewards = self.run_env(self.hard_env)
        hard_mean_rew = safemean(hard_rewards)
        
        full_rewards = self.run_env(self.full_env)
        full_mean_rew = safemean(full_rewards)

        return easy_mean_rew, hard_mean_rew, full_mean_rew
    
    def run_env(self, env) -> np.ndarray:
        states = self._model.adr_initial_state
        obs = env.reset()
        dones = [False]

        epinfos = []

        # For n in range number of steps
        n_completed = 0
        while n_completed < self._n_trajectories:
            # Given observations, get action value
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, _, states, _ = self._model.adr_step(obs, S=states, M=dones)

            # Take actions in env and look the results
            # Info contains a ton of useful information
            obs[:], rewards, dones, infos = env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            if dones[0]:
                n_completed += 1

        # Get the mean of the returns for these trajectories and add them to the buffer.
        episode_rewards = np.array([epinfo['r'] for epinfo in epinfos])

        return episode_rewards


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
