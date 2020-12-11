# Import everything
from collections import deque
import numpy as np
from baselines import logger
from baselines.ppo2.runner import Runner
from baselines.common.policies import build_policy
from baselines.common.vec_env.vec_env import VecEnv
from baselines_adr.adr_model import ADRModel


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def safestd(xs):
    return np.nan if len(xs) == 0 else np.std(xs)


def test(network,
         test_env: VecEnv,
         n_steps: int = 2048,
         ent_coef: float = 0.,
         vf_coef: float = .5,
         max_grad_norm: float = .5,
         gamma: float = .99,
         lmbda: float = .95,
         n_minibatches: int = 1,
         load_path: str = None,
         model_fn= None,
         mpi_rank_weight: int = 1,
         comm=None,
         **network_kwargs):

    # Load models
    policy = build_policy(test_env, network, **network_kwargs)

    # Get the nb of env
    nenvs = test_env.num_envs

    # Get state_space and action_space
    ob_space = test_env.observation_space
    ac_space = test_env.action_space

    # Calculate the batch_size
    nbatch = nenvs * n_steps
    nbatch_train = nbatch // n_minibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        #model_fn = Model
        model_fn = ADRModel

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, comm=comm,
                     mpi_rank_weight=mpi_rank_weight)

    model.load(load_path)
    runner = Runner(env=test_env, model=model, nsteps=n_steps, gamma=gamma, lam=lmbda)
    epinfobuf = deque(maxlen=100)

    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
    epinfobuf.extend(epinfos)

    #get reward stats
    eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
    eprewstd = safestd([epinfo['r'] for epinfo in epinfobuf])
    logger.logkv('eprewmean', eprewmean)
    logger.logkv('eprewstd', eprewstd)
    return eprewmean, eprewstd
