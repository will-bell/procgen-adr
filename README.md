# procgen-adr

Procgen ADR is a python implementation of Automatic Domain Randomization by [Open AI](https://openai.com/blog/solving-rubiks-cube/)

Team Members include: William Bell (wjbell@bu.edu), Tu Timmy Hoang (hoangt@bu.edu), David McIntyre (dpmc@bu.edu), Danny Trinh (djtrinh@bu.edu)

## Installation

We created a fork of OpenAI's baselines which have useful reinforcement learning modules. We specifically use PPO and Impala CNN.
In the fork we updated the repo to be compatible with TensorFlow 2.0.0.
Install from source:
[https://github.com/tuthoang/baselines](https://github.com/tuthoang/baselines)

We also forked OpenAI's procgen in order to make customizable environments. Install from source:
[https://github.com/will-bell/procgen](https://github.com/will-bell/procgen)


## Usage

```bash
python -m baselines_adr.train --env_name dc_bossfight --n_train_envs 128 --n_training_steps 200000000 --log_dir ./recurr  --recur True
```

This will train a recurrent policy on our game, dc_bossfight on 128 parallal environments over 2 million training steps. Models and progress will be periodically saved in /adr_experiments/{some unique identifier}/recurr.