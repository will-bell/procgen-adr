from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize, VecEnvWrapper, VecFrameStack, DummyVecEnv, SubprocVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from procgen import ProcgenEnv, ProcgenGym3Env
from procgen.domains import datetime_name
from mpi4py import MPI
from baselines.common.policies import build_policy
from baselines_adr.adr_model import ADRModel
from baselines.common.models import build_impala_cnn
from baselines import logger

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


import os, glob
import numpy as np
import gym
from gym3 import VideoRecorderWrapper

video_length=1000
video_interval = 1
test_domain_config_path = 'test-config.json'
env_name = 'dc_bossfight'
training_env = ProcgenEnv(num_envs=1, env_name=env_name, domain_config_path=str(test_domain_config_path), render_mode="rgb_array")
training_env = VecExtractDictObs(training_env, "rgb")
training_env = VecMonitor(venv=training_env, filename=None, keep_buf=100)
training_env = VecNormalize(venv=training_env, ob=False)
# training_env = VideoRecorderWrapper(training_env, directory ='./recordings', info_key='rgb')

training_env = VecVideoRecorder(training_env, './recordings', record_video_trigger=lambda x: x % video_interval == 0, video_length=video_length)

# with tf.Session() as sess:

def conv_fn(x):
    return build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

network = conv_fn

policy = build_policy(training_env, network)

# Get the nb of env
nenvs = training_env.num_envs

# Get state_space and action_space
ob_space = training_env.observation_space
ac_space = training_env.action_space



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
# Calculate the batch_size
nbatch = nenvs * n_steps
nbatch_train = nbatch // n_minibatches
# Instantiate the model object (that creates act_model and train_model)
# if model_fn is None:
model_fn = ADRModel

comm = MPI.COMM_WORLD
mpi_rank_weight = 0
model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, comm=comm,
                    mpi_rank_weight=mpi_rank_weight)

load_path = '000002400.ckpt'
if load_path is not None:
    model.load(load_path)
                
obs = training_env.reset()
dones = [False]
states = model.initial_state
import numpy as np
step = 0

rew = []
for _ in range(video_interval + video_length + 1):
    actions, values, states, _ = model.step(obs, S=states, M=dones)
    obs[:], rewards, dones, infos = training_env.step(actions)
    rew.append(rewards)
    step+=1
    print(f"Steps: {step}")
    training_env.render()
    if dones[0]:
        break
    
print(np.mean(rew))
training_env.close()

    # recorded_video = glob.glob(os.path.join('./recordings', "*.mp4"))
