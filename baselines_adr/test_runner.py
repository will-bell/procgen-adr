import copy
import pathlib
from typing import List

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
        self.model = model
        self._n_trajectories = n_trajectories

        # Initialize the environment
        ez_config_path = config_dir / 'test_easy_config.json'
        ez_config = copy.copy(BossfightEasyConfig)
        ez_config.to_json(ez_config_path)
        ez_env = ProcgenEnv(num_envs=1, env_name=str(ez_config.game), domain_config_path=str(ez_config_path))
        ez_env = VecExtractDictObs(ez_env, "rgb")
        ez_env = VecMonitor(venv=ez_env, filename=None, keep_buf=100)
        self.ez_env = VecNormalize(venv=ez_env, ob=False)

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

        self.states = self.model.adr_initial_state

    def run(self):
        ez_rew = self.run_env(self.ez_env)
        hard_rew = self.run_env(self.hard_env)
        full_rew = self.run_env(self.full_env)

        return ez_rew, hard_rew, full_rew

    def run_env(self, env):
        # Here, we init the lists that will contain the mb of experiences
        mb_rewards, mb_values, mb_dones = [], [], []
        epinfos = []
        states = self.model.adr_initial_state

        # For n in range number of steps
        n_completed = 0
        _obs = env.reset()
        _dones = [False]
        while n_completed < self._n_trajectories:
            actions, values, states, _ = self.model.adr_step(_obs, S=states, M=_dones)
            mb_values.append(values)
            mb_dones.append(_dones)

            # Take actions in env and look the results
            # Info contains a ton of useful information
            _obs[:], rewards, _dones, infos = env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

            if _dones[0]:
                n_completed += 1
        return safemean(np.array(mb_rewards))


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


