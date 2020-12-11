import tensorflow.compat.v1 as tf
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines import logger
from baselines.common.models import build_impala_cnn, cnn_lstm
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from test import test
import argparse
import os


def test_fn(env_name, num_envs, config_path, load_path):
    test_config_path = os.path.join(os.getcwd(), "procgen-adr", config_path)
    test_env = ProcgenEnv(num_envs=num_envs, env_name=env_name, domain_config_path=test_config_path, render_mode="rgb_array")
    test_env = VecExtractDictObs(test_env, "rgb")
    test_env = VecMonitor(venv=test_env, filename=None, keep_buf=100)
    test_env = VecNormalize(venv=test_env, ob=False)

    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    recur = True
    if recur:
        logger.info("Using CNN LSTM")
        conv_fn = cnn_lstm(nlstm=256, conv_fn=conv_fn)

    mean, std = test(conv_fn, test_env, load_path=load_path)
    sess.close()
    return mean, std

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='dc_bossfight')
    parser.add_argument('--num_envs', type=int, default=10)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=200_000_000)

    args = parser.parse_args()
    test_domain_config_path = os.path.join(os.getcwd(), "procgen-adr", "easy_config.json")
    load_path = 'procgen-adr/models/model_adr.ckpt'

    mean, std = test_fn(args.env_name, args.num_envs, test_domain_config_path, load_path)
    print("MEAN is %f" % mean)
    print("STD is %f" % std)


if __name__ == '__main__':
    main()