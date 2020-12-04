import numpy as np
from baselines.common.runners import AbstractEnvRunner
from procgen.domains import DomainConfig, BossfightHardConfig

class TestRunner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, model, nsteps, gamma, lam, config_paths: dict, n_trajectories: int):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        
        self._n_trajectories = n_trajectories

        # Initialize the environment
        ez_config = DomainConfig.from_json(config_paths['ez_env'])
        ez_env = ProcgenEnv(num_envs=1, env_name=str(ez_config.game),
                         domain_config_path=str(config_paths['ez_env']))
        ez_env = VecExtractDictObs(ez_env, "rgb")
        ez_env = VecMonitor(venv=ez_env, filename=None, keep_buf=100)
        self.ez_env = VecNormalize(venv=ez_env, ob=False)
        
        hard_config = DomainConfig.from_json(config_paths['hard_env'])
        hard_env = ProcgenEnv(num_envs=1, env_name=str(hard_config.game),
                         domain_config_path=str(config_paths['hard_env']))
        hard_env = VecExtractDictObs(hard_env, "rgb")
        hard_env = VecMonitor(venv=hard_env, filename=None, keep_buf=100)
        self.hard_env = VecNormalize(venv=hard_env, ob=False)
        
        full_config = DomainConfig.from_json(config_paths['full_env'])
        full_env = ProcgenEnv(num_envs=1, env_name=str(full_config.game),
                         domain_config_path=str(config_paths['full_env']))
        full_env = VecExtractDictObs(full_env, "rgb")
        full_env = VecMonitor(venv=full_env, filename=None, keep_buf=100)
        self.full_env = VecNormalize(venv=full_env, ob=False)
        

        self.states = self.model.adr_initial_state

    def run(self):
        ez_rew = self.run_env(self.ez_env)
        hard_rew = self.run_env(self.hard_rew)
        full_rew = self.run_env(self.full_rew)

        return ez_rew, hard_rew, full_rew

    def run_env(self, env):

        # Here, we init the lists that will contain the mb of experiences
        mb_rewards, mb_values, mb_dones = [], [], []
        epinfos = []
        states = None

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


