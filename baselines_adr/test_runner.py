import copy
import pathlib
from typing import List
from collections import deque

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
    def __init__(self, model, config_dir: pathlib.Path, n_trajectories: int, tunable_params: List[EnvironmentParameter], gamma, lam):
        self.model = model
        self._n_trajectories = n_trajectories
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

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
        epinfobuf = deque(maxlen=100)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.run_env(self.ez_env)
        epinfobuf.extend(epinfos)
        ez_rew = safemean([epinfo['r'] for epinfo in epinfobuf])
        
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.run_env(self.hard_env)
        epinfobuf.extend(epinfos)
        hard_rew = safemean([epinfo['r'] for epinfo in epinfobuf])
        
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.run_env(self.full_env)
        epinfobuf.extend(epinfos)
        full_rew = safemean([epinfo['r'] for epinfo in epinfobuf])
        return ez_rew, hard_rew, full_rew
    
    def run_env(self, env):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = copy.copy(self.model.adr_initial_state)
        states = copy.copy(self.model.adr_initial_state)
        epinfos = []
        # For n in range number of steps
        n_completed = 0
        obs = env.reset()
        dones = [False]
        while n_completed < self._n_trajectories:
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, neglogpacs = self.model.adr_step(obs, S=states, M=dones)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs[:], rewards, dones, infos = env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            if dones[0]:
                n_completed += 1
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.adr_value(obs, S=states, M=dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        nsteps = len(mb_dones)
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


