import pathlib
import random
from typing import Union, List, Tuple, Optional, Dict
import os
import os.path as osp
import numpy as np
import pandas as pd
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from procgen import ProcgenEnv
from procgen.domains import DomainConfig, BossfightDomainConfig
from baselines import logger

Number = Union[int, float]

DEFAULT_DOMAIN_CONFIGS = {
    'dc_bossfight': BossfightDomainConfig()
}


class EnvironmentParameter:

    lower_bound: Number
    upper_bound: Number

    delta: Number

    def __init__(self, name: str, initial_bounds: Tuple[Number, Number], clip_bounds: Tuple[Number, Number],
                 delta: Number, discrete: bool):
        self.name = name
        self.lower_bound, self.upper_bound = initial_bounds
        self.initial_lower_bound, self.initial_upper_bound = initial_bounds
        self.clip_lower_bound, self.clip_upper_bound = clip_bounds
        self.discrete = discrete
        self.delta = delta
        
    def increase_lower_bound(self):
        self.lower_bound = min(self.lower_bound + self.delta, self.initial_lower_bound)

    def decrease_lower_bound(self):
        self.lower_bound = max(self.lower_bound - self.delta, self.clip_lower_bound)

    def increase_upper_bound(self):
        self.upper_bound = min(self.upper_bound + self.delta, self.clip_upper_bound)

    def decrease_upper_bound(self):
        self.upper_bound = max(self.upper_bound - self.delta, self.initial_upper_bound)


DEFAULT_TUNABLE_PARAMS = {
    'dc_bossfight': [
        EnvironmentParameter(name='n_rounds', initial_bounds=(1, 1), clip_bounds=(1, 10), delta=1, discrete=True),
        EnvironmentParameter(name='n_barriers', initial_bounds=(1, 1), clip_bounds=(1, 5), delta=1, discrete=True),
        EnvironmentParameter(name='boss_round_health', initial_bounds=(1, 1), clip_bounds=(1, 10), delta=1, discrete=True),
        EnvironmentParameter(name='boss_invulnerable_duration', initial_bounds=(1, 1), clip_bounds=(1, 10), delta=1, discrete=True),
        EnvironmentParameter(name='boss_bullet_velocity', initial_bounds=(.1, .1), clip_bounds=(.1, 1.), delta=.1, discrete=False),
    ]
}


class PerformanceBuffer:

    def __init__(self):
        self._buffer = []

    def push_back(self, value: float):
        self._buffer.append(value)

    def is_full(self, size: int) -> bool:
        return len(self._buffer) >= size

    def calculate_average_performance(self) -> float:
        buffer = np.array(self._buffer)
        self.clear()
        return np.mean(buffer).item()

    def clear(self):
        self._buffer = []


class ADRConfig:

    n_eval_trajectories: int

    max_buffer_size: int

    gamma: float

    lmbda: float

    performance_thresholds: Tuple[float, float]

    upper_sample_prob: float

    use_gae: bool

    def __init__(self,
                 n_eval_trajectories: int = 10,
                 max_buffer_size: int = 7,
                 gamma: float = .999,
                 lmbda: float = .95,
                 performance_thresholds: Tuple[float, float] = (2.5, 6.),
                 upper_sample_prob: float = .8,
                 use_gae: bool = True):

        self.n_eval_trajectories = n_eval_trajectories
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.performance_thresholds = performance_thresholds
        self.upper_sample_prob = upper_sample_prob
        self.use_gae = use_gae


class ParameterRunner:

    _gamma: float
    _lambda: float

    _env_parameter: EnvironmentParameter

    _n_trajectories: int
    _max_buffer_size: int
    _train_config_path: pathlib.Path

    _param_name: str

    _boundary_config_path: pathlib.Path
    _boundary_config: DomainConfig

    _upper_performance_buffer: PerformanceBuffer
    _lower_performance_buffer: PerformanceBuffer

    def __init__(self, model, train_config_path: Union[str, pathlib.Path], env_parameter: EnvironmentParameter,
                 adr_config: ADRConfig):

        self._model = model    # Model being evaluated
        self._gamma = adr_config.gamma    # Discount rate
        self._lambda = adr_config.lmbda   # Lambda used in GAE (General Advantage Estimation)

        self._env_parameter = env_parameter
        self._param_name = self._env_parameter.name

        self._max_buffer_size = adr_config.max_buffer_size
        self._n_trajectories = adr_config.n_eval_trajectories
        self._upper_sample_prob = adr_config.upper_sample_prob

        self._train_config_path = pathlib.Path(train_config_path)
        config_dir = self._train_config_path.parent
        config_name = self._param_name + '_adr_eval_config.json'

        # Initialize the config for the evaluation environment
        # This config will be updated regularly throughout training. When we boundary sample this environment's
        # parameter, the config will be modified to set the parameter to the selected boundary before running a number
        # of trajectories.
        self._boundary_config = DomainConfig.from_json(self._train_config_path)
        self._boundary_config_path = config_dir / config_name
        self._boundary_config.to_json(self._boundary_config_path)

        # Initialize the environment
        env = ProcgenEnv(num_envs=1, env_name=str(self._boundary_config.game),
                         domain_config_path=str(self._boundary_config_path))
        env = VecExtractDictObs(env, "rgb")
        env = VecMonitor(venv=env, filename=None, keep_buf=100)
        self._env = VecNormalize(venv=env, ob=False)

        # Initialize the performance buffers
        self._upper_performance_buffer, self._lower_performance_buffer = PerformanceBuffer(), PerformanceBuffer()

        self._states = model.adr_initial_state
        self._obs = self._env.reset()
        self._dones = [False]
        self.hidden_state = None

    def evaluate_performance(self) -> Optional[Tuple[float, bool]]:
        """Main method for running the ADR algorithm

        Returns:

        """
        # Load the current training config to get any changes to the environment parameters
        updated_train_config = DomainConfig.from_json(self._train_config_path)
        updated_params = updated_train_config.parameters

        x = random.uniform(0., 1.)
        if x < self._upper_sample_prob:
            lower = False
            value = self._env_parameter.upper_bound
            buffer = self._upper_performance_buffer
        else:
            lower = True
            value = self._env_parameter.lower_bound
            buffer = self._lower_performance_buffer

        updated_params['min_' + self._param_name] = value
        updated_params['max_' + self._param_name] = value
        self._boundary_config.update_parameters(updated_params, cache=False)

        self._generate_trajectories(buffer)

        if buffer.is_full(self._max_buffer_size):
            performance = buffer.calculate_average_performance()
            return performance, lower

        return None

    def increase_lower_bound(self) -> Number:
        self._env_parameter.increase_lower_bound()
        return self._env_parameter.lower_bound

    def decrease_lower_bound(self) -> Number:
        self._env_parameter.decrease_lower_bound()
        return self._env_parameter.lower_bound

    def increase_upper_bound(self) -> Number:
        self._env_parameter.increase_upper_bound()
        return self._env_parameter.upper_bound

    def decrease_upper_bound(self) -> Number:
        self._env_parameter.decrease_upper_bound()
        return self._env_parameter.upper_bound

    def _generate_trajectories(self, buffer: PerformanceBuffer):
        # Here, we init the lists that will contain the mb of experiences
        mb_rewards, mb_values, mb_dones = [], [], []
        epinfos = []

        # For n in range number of steps
        n_completed = 0
        while n_completed < self._n_trajectories:
            # Given observations, get action value
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self._states, _ = self._model.adr_step(self._obs, S=self._states, M=self._dones)
            mb_values.append(values)
            mb_dones.append(self._dones)

            # Take actions in env and look the results
            # Info contains a ton of useful information
            self._obs[:], rewards, self._dones, infos = self._env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

            if self._dones[0]:
                n_completed += 1

        # batch of steps to batch of rollouts
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self._model.adr_value(self._obs, S=self._states, M=self._dones)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        n_steps = len(mb_dones)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - self._dones
                nextvalues = last_values

            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]

            delta = mb_rewards[t] + self._gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self._gamma * self._lambda * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        # Get the mean of the returns for these trajectories and add them to the buffer.
        buffer.push_back(np.mean(mb_returns, axis=0).item())

    def get_clip_boundaries(self):
        return self._env_parameter.clip_lower_bound, self._env_parameter.clip_upper_bound

    def get_values(self):
        return self._env_parameter.lower_bound, self._env_parameter.upper_bound

class ADRRunner:
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    _tunable_param_names: List[str]

    _param_runners: Dict[str, ParameterRunner]

    def __init__(self, model, initial_domain_config: DomainConfig, tunable_parameters: List[EnvironmentParameter],
                 adr_config: ADRConfig = None):

        self.model = model
        self._train_domain_config = initial_domain_config
        train_config_path = initial_domain_config.path

        adr_config = ADRConfig() if adr_config is None else adr_config

        self._tunable_param_names = []
        self._param_runners = {}
        for param in tunable_parameters:
            self._tunable_param_names.append(param.name)
            self._param_runners[param.name] = ParameterRunner(model, train_config_path, param, adr_config)
        self._n_tunable_params = len(self._tunable_param_names)

        self._low_threshold, self._high_threshold = adr_config.performance_thresholds
        
        self.filename = f'adr_log.csv'
        self.list_changes = []

    def run(self, update_iter):
        
        # Randomly select a parameter to boundary sample
        param_idx = random.randint(0, self._n_tunable_params - 1)
        param_name = self._tunable_param_names[param_idx]

        # Get the environment for the selected parameter then evaluate the policy within it. This will boundary sample
        # the selected parameter and generate a number of trajectories with the upper/lower boundary to calculate its
        # performance in the environment.
        param_runner = self._param_runners[param_name]
        info = param_runner.evaluate_performance()
        old_value_low, old_value_upper = param_runner.get_values()
        # If we get something back, then the performance buffer for either the lower or upper boundary of the parameter
        # is filled. Update the parameter according to the set thresholds.
        if info:
            performance, lower = info
            new_value = None
            selected_bound = None
            if lower:  # Updating the lower boundary according to set performance thresholds
                prefix = 'min_'
                selected_bound = 'lower'
                old_value, other_value = old_value_low, old_value_upper

                if performance >= self._high_threshold:
                    # TODO: Log change
                    new_value = param_runner.increase_lower_bound()
                elif performance <= self._low_threshold:
                    # TODO: Log change
                    new_value = param_runner.decrease_lower_bound()

            else:  # Updating the upper boundary according to set performance thresholds
                prefix = 'max_'
                selected_bound = 'upper'
                old_value, other_value = old_value_upper, old_value_low

                if performance >= self._high_threshold:
                    # TODO: Log change
                    new_value = param_runner.increase_upper_bound()
                elif performance <= self._low_threshold:
                    # TODO: Log change
                    new_value = param_runner.decrease_upper_bound()

            # Update the config for the training environment with the new value for the boundary-sampled parameter if
            # we got a new value.
            if new_value is not None:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i.ckpt' % update_iter)
                print('Saving to', savepath)
                self.model.save(savepath)
                config_savepath = osp.join(checkdir, '%.5i.json' % update_iter)
                self._train_domain_config.to_json(config_savepath)

                logger.info(f"Saving model to {savepath}")
                logger.info(f"Saving config to {config_savepath}")

                entropy = self.adr_entropy()
                logger.info(f"ADR Entropy to {entropy}")
                
                min_clip, max_clip = param_runner.get_clip_boundaries()
                self.list_changes.append([update_iter, prefix, param_name, selected_bound, performance, 
                                        old_value, new_value, other_value, entropy, self._low_threshold, self._high_threshold,
                                        min_clip, max_clip])
                
                log_df = pd.DataFrame(self.list_changes, 
                                       columns=['update_iter', 'prefix', 'param_name', 'selected_bound', 'performance', 'old_value', 
                                            'new_value', 'other_bound_value', 'adr_entropy', 'low_perf_thresh', 'high_perf_thresh', 
                                            'min_clip', 'max_clip'])
                log_df_savepath = os.path.join(logger.get_dir(), self.filename)
                log_df.to_csv(log_df_savepath, encoding='utf-8', index=False)


                self._train_domain_config.update_parameters({prefix + param_name: new_value})
                
    def adr_entropy(self):
        """ Calculate ADR Entropy

        Returns:
            float: entropy =  1/d \sum_{i=1}^{d} log(phi_ih - phi_il)
        """

        differences = []
        for param_runner in self._param_runners.values():
            phi_l, phi_h = param_runner.get_values()

            differences.append(np.log(phi_h - phi_l))

        entropy = np.mean(differences)
        return entropy