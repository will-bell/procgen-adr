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


## Files Description
### Baselines ADR
- adr_model.py - contains the model that is used to generate actions inside the environment loop
- adr_runner.py - contains all the necessary classes and configs as well as the ParameterRunner and ADRRunner that make the ADR algorithm possible inside the training loop
- ppo2_adr.py - training loop that runs ADR and generates data for updating policy with PPO
- test_runner.py - runner for evaluating the model on the three environments (easy, hard, full ADR range) during training
- train.py - command line script for running the training algorithm

### Test Agent
- test.py - contains functions to play test environmnets and return results
- procgen_test.py - runs test environment loaded from trained model on specified environment config
- plot_results.ipynb - simple notebook to plot and compare traning results of different models
- models/ - model checkpoints used for evaluating performance
- configs/ - environmnetal configurations used for evaluating performance
