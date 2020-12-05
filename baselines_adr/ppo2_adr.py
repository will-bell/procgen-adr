import os
import os.path as osp
import pathlib
import time
from collections import deque
from random import uniform
from typing import List

import numpy as np
from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.vec_env.vec_env import VecEnv
from baselines.ppo2.runner import Runner
from mpi4py import MPI
from procgen.domains import DomainConfig

from baselines_adr.adr_model import ADRModel
from baselines_adr.adr_runner import ADRRunner, ADRConfig, EnvironmentParameter
from baselines_adr.test_runner import TestRunner


def constfn(val):
    def f(_):
        return val
    return f


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def learn(network,
          training_env: VecEnv,
          n_training_steps: int,
          config_dir: pathlib.Path,
          adr_config: ADRConfig,
          train_domain_config: DomainConfig,
          tunable_parameters: List[EnvironmentParameter],
          eval_env: VecEnv = None,
          seed: int = None,
          n_steps: int = 2048,
          ent_coef: float = 0.,
          lr: float = 3e-4,
          vf_coef: float = .5,
          max_grad_norm: float = .5,
          gamma: float = .99,
          lmbda: float = .95,
          log_interval: int = 10,
          save_interval: int = 0,
          n_minibatches: int = 4,
          n_optepochs: int = 4,
          clip_range: float = .2,
          load_path: str = None,
          model_fn: ADRModel = None,
          mpi_rank_weight: int = 1,
          comm=None,
          **network_kwargs):

    """
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347) combined with ADR

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn,
                                      cnn_small, conv_only - see baselines.common/models.py for full list) specifying
                                      the standard network architecture, or a function that takes tensorflow tensor as
                                      input and returns tuple (output_tensor, extra_feed) where output tensor is the
                                      last network layer output, extra_feed is None for feed-forward neural nets, and
                                      extra_feed is a dictionary describing how to feed state into the network for
                                      recurrent neural nets. See common/models.py/lstm for more details on using
                                      recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                             The environments produced by gym.make can be wrapped using
                                             baselines.common.vec_env.DummyVecEnv class.

    n_steps: int                       number of steps of the vectorized environment per update (i.e. batch size is
                                      n_steps * nenv where nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of
                                      the training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lmbda: float                      advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    n_minibatches: int                number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    n_optepochs: int                   number of training epochs per update

    clip_range: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of
                                      the training and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/
                                      policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    """
    set_global_seeds(seed)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)

    if isinstance(clip_range, float):
        clip_range = constfn(clip_range)
    else:
        assert callable(clip_range)

    n_training_steps = int(n_training_steps)

    policy = build_policy(training_env, network, **network_kwargs)

    # Get the nb of env
    nenvs = training_env.num_envs

    # Get state_space and action_space
    ob_space = training_env.observation_space
    ac_space = training_env.action_space

    # Calculate the batch_size
    nbatch = nenvs * n_steps
    nbatch_train = nbatch // n_minibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        model_fn = ADRModel

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, comm=comm,
                     mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object for generating training data
    runner = Runner(env=training_env, model=model, nsteps=n_steps, gamma=gamma, lam=lmbda)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=n_steps, gamma=gamma, lam=lmbda)
    else:
        eval_runner = None

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)
    else:
        eval_epinfobuf = None

    # Instantiate the runner object for the ADR algorithm
    adr_runner = ADRRunner(model, train_domain_config, tunable_parameters, adr_config)

    # Instantiate the runner object for the test environments
    test_runner = TestRunner(model, config_dir, adr_config.n_eval_trajectories, tunable_parameters)

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = n_training_steps // nbatch
    update = 1
    while update <= nupdates:
        # At the top of the training loop,
        run_adr = uniform(0., 1.) < .5
        if run_adr:
            adr_runner.run(update)

        else:
            assert nbatch % n_minibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (update - 1.0) / nupdates

            # Calculate the learning rate
            lrnow = lr(frac)

            # Calculate the clip_range
            clip_rangenow = clip_range(frac)

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Stepping environment...')

            # Get minibatch and add to the information buffers
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
            epinfobuf.extend(epinfos)
            if eval_runner is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, \
                    eval_epinfos = eval_runner.run()
                if eval_epinfobuf is not None:
                    eval_epinfobuf.extend(eval_epinfos)

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Done.')

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            if states is None:  # non-recurrent version
                # Index of each element of batch_size
                # Create the indices array
                inds = np.arange(nbatch)
                for _ in range(n_optepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, clip_rangenow, *slices))

            else:  # recurrent version
                assert nenvs % n_minibatches == 0
                envsperbatch = nenvs // n_minibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * n_steps).reshape(nenvs, n_steps)
                for _ in range(n_optepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, clip_rangenow, *slices, mbstates))

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)

            # End timer/
            tnow = time.perf_counter()

            # Calculate the fps (frame per second)
            fps = int(nbatch / (tnow - tstart))

            if update % log_interval == 0 or update == 1:
                easy_rew, hard_rew, full_rew = test_runner.run()

                # Calculates if value function is a good predictor of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, returns)
                logger.logkv("misc/serial_timesteps", update*n_steps)
                logger.logkv("misc/nupdates", update)
                logger.logkv("misc/total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("misc/explained_variance", float(ev))
                logger.logkv('train_eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('train_eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('eprewmean_easy', easy_rew)
                logger.logkv('eprewmean_hard', hard_rew)
                logger.logkv('eprewmean_full', full_rew)

                if eval_env is not None:
                    logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                    logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))

                logger.logkv('misc/time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv('loss/' + lossname, lossval)

                logger.dumpkvs()

            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i' % update)
                print('Saving to', savepath)
                model.save(savepath)

            update += 1

    return model
