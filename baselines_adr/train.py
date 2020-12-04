import os, sys # Windows workaround to importing baselines_adr
sys.path.append(os.path.curdir)

import argparse
import pathlib

import tensorflow.compat.v1 as tf
from baselines import logger
from baselines.common.models import build_impala_cnn, cnn_lstm, impala_cnn_lstm, cnn_lnlstm
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from mpi4py import MPI
from procgen import ProcgenEnv
from procgen.domains import datetime_name

from baselines_adr import ppo2_adr
from baselines_adr.adr_runner import ADRConfig, DEFAULT_DOMAIN_CONFIGS, DEFAULT_TUNABLE_PARAMS


def train_fn(env_name: str,
             num_train_envs: int,
             n_training_steps: int,
             adr_config: ADRConfig = None,
             experiment_dir: str = None,
             tunable_params_config_path: str = None,
             log_dir: str = None,
             is_test_worker: bool = False,
             comm=None,
             save_interval: int = 1000,
             log_interval: int = 1,
             recur: bool = False):

    # Get the default ADR config if none is provided
    adr_config = ADRConfig() if adr_config is None else adr_config

    # Set up the experiment directory for this run. This will contain everything, from the domain configs for the
    # training environment and ADR evaluation environments to the logs. If the directory path is not provided, then
    # we'll make one an use the date-time-name to make it unique
    if experiment_dir is None:
        experiment_dir = pathlib.Path().absolute() / 'adr_experiments' / ('experiment-' + datetime_name())
        experiment_dir.mkdir(parents=True, exist_ok=False)
    else:
        experiment_dir = pathlib.Path(experiment_dir)

    # Make a config directory within the experiment directory to hold the domain configs
    config_dir = experiment_dir / 'domain_configs'
    config_dir.mkdir(parents=True, exist_ok=False)

    # Load the tunable parameters from a config file if it is provided, otherwise get the default for the given game.
    if tunable_params_config_path is None:
        try:
            tunable_params = DEFAULT_TUNABLE_PARAMS[env_name]
        except KeyError:
            raise KeyError(f'No default tunable parameters exist for {env_name}')
    else:
        raise NotImplemented('Currently no way to load tunable parameters from a configuration file')

    # Make a default config for the given game...
    train_domain_config_path = config_dir / 'train_config.json'
    try:
        train_domain_config = DEFAULT_DOMAIN_CONFIGS[env_name]
        train_domain_config.to_json(train_domain_config_path)
    except KeyError:
        raise KeyError(f'No default config exists for {env_name}')

    # ...then load the initial bounds for the tunable parameters into the config.
    params = {}
    for param in tunable_params:
        params['min_' + param.name] = param.lower_bound
        params['max_' + param.name] = param.upper_bound
    train_domain_config.update_parameters(params, cache=False)

    # Configure the logger if we are given a log directory
    if log_dir is not None:
        log_dir = experiment_dir / log_dir
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=str(log_dir), format_strs=format_strs)

    logger.info(f'env_name: {env_name}')
    logger.info(f'num_train_envs: {num_train_envs}')
    logger.info(f'n_training_steps: {n_training_steps}')
    logger.info(f'experiment_dir: {experiment_dir}')
    logger.info(f'tunable_params_config_path: {tunable_params_config_path}')
    logger.info(f'log_dir: {log_dir}')
    logger.info(f'save_interval: {save_interval}')

    n_steps = 256
    ent_coef = .01
    lr = 5e-4
    vf_coef = .5
    max_grad_norm = .5
    gamma = .999
    lmbda = .95
    n_minibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1

    logger.info('creating environment')
    training_env = ProcgenEnv(num_envs=num_train_envs, env_name=env_name, domain_config_path=str(train_domain_config_path))
    training_env = VecExtractDictObs(training_env, "rgb")
    training_env = VecMonitor(venv=training_env, filename=None, keep_buf=100)
    training_env = VecNormalize(venv=training_env, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.__enter__()

    def conv_fn(x):
        return build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    if recur:
        logger.info("Using CNN LSTM")
        conv_fn = cnn_lstm(nlstm=256, conv_fn=conv_fn)
    
    logger.info('training')
    ppo2_adr.learn(
        conv_fn,
        training_env,
        n_training_steps,
        config_dir,
        adr_config,
        train_domain_config,
        tunable_params,
        n_steps=n_steps,
        ent_coef=ent_coef,
        lr=lr,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        lmbda=lmbda,
        log_interval=log_interval,
        save_interval=save_interval,
        n_minibatches=n_minibatches,
        n_optepochs=ppo_epochs,
        clip_range=clip_range,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping)


def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='dc_bossfight')
    parser.add_argument('--n_train_envs', type=int, default=2)
    parser.add_argument('--n_training_steps', type=int, default=200000000)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--recur', type=bool, default=False)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(
        args.env_name,
        args.n_train_envs,
        args.n_training_steps,
        is_test_worker=is_test_worker,
        comm=comm,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        recur=args.recur)


if __name__ == '__main__':
    main()
