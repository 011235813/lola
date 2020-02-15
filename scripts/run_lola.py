"""The main scripts for running different scenarios."""

import click
import time

from lola import logger

from lola.envs import *
import sys
sys.path.append('../../lio/env/')
from room_symmetric_lola import EscapeRoom
from room_asymmetric_lola import EscapeRoomAsym

@click.command()
# Experiment parameters
@click.option("--exp_name", type=str, default="IPD",
              help="Name of the experiment (and correspondingly environment).")
@click.option("--num_episodes", type=int, default=None,
              help="Number of episodes.")
@click.option("--trace_length", type=int, default=None,
              help="Lenght of the traces.")
@click.option("--exact/--no-exact", default=True,
              help="Whether to run the exact version of LOLA.")
@click.option("--pseudo/--no-pseudo", default=False,
              help="Only used with exact version of LOLA.")
@click.option("--grid_size", type=int, default=3,
              help="Grid size of the coin game (used only for coin game).")
@click.option("--trials", type=int, default=2, help="Number of trials.")
@click.option("--n_agents", type=int, default=2, help="Number of agents.")

# Learning parameters
@click.option("--lola/--no-lola", default=True,
              help="Add the crazy LOLA corrections to the computation.")
@click.option("--opp_model/--no-opp_model", default=False,
              help="Whether to model opponent or use true parameters "
                   "(use only for coin game).")
@click.option("--mem_efficient/--no-mem_efficient", default=True,
              help="Use a more memory efficient corrections method.")
@click.option("--lr", type=float, default=None,
              help="Learning rate for Adam optimizer.")
@click.option("--lr_correction", type=float, default=1,
              help="Learning rate for corrections.")
@click.option("--batch_size", type=int, default=None,
              help="Number of episodes to optimize at the same time.")
@click.option("--bs_mul", type=int, default=1,
              help="Number of episodes to optimize at the same time")

# Policy parameters
@click.option("--simple_net/--no-simple_net", default=True,
              help="Use a simple policy (only for IPD and IMP).")
@click.option("--hidden", type=int, default=32,
              help="Size of the hidden layer.")
@click.option("--num_units", type=int, default=64,
              help="Number of units in the MLP.")
@click.option("--reg", type=float, default=0.,
              help="Regularization parameter.")
@click.option("--gamma", type=float, default=None,
              help="Discount factor.")

def main(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         trials, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
         num_units, reg, gamma, lola, opp_model, mem_efficient, n_agents):
    # Sanity
    assert exp_name in {"CoinGame", "IPD", "IMP", "escape-room",
                        "asym-escape-room"}

    # Resolve default parameters
    if exact:
        num_episodes = 50 if num_episodes is None else num_episodes
        trace_length = 200 if trace_length is None else trace_length
        lr = 1. if lr is None else lr
    elif exp_name in {"IPD", "IMP"}:
        num_episodes = 600000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 1. if lr is None else lr
    elif exp_name == "CoinGame":
        num_episodes = 100000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 0.005 if lr is None else lr
    elif exp_name == "escape-room" or exp_name == "asym-escape-room":
        num_episodes = 50000 if num_episodes is None else num_episodes
        trace_length = 5 if trace_length is None else trace_length
        batch_size = 50 if batch_size is None else batch_size
        lr = 1. if lr is None else lr

    # Import the right training function
    if exact:
        assert exp_name != "CoinGame", "Can't run CoinGame with --exact."
        def run(env):
            from lola.train_exact import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  simple_net=simple_net,
                  corrections=lola,
                  pseudo=pseudo,
                  num_hidden=hidden,
                  reg=reg,
                  lr=lr,
                  lr_correction=lr_correction,
                  gamma=gamma)
    elif exp_name in {"IPD", "IMP"}:
        def run(env):
            from lola.train_pg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  gamma=gamma,
                  set_zero=0,
                  lr=lr,
                  corrections=lola,
                  simple_net=simple_net,
                  hidden=hidden,
                  mem_efficient=mem_efficient)
    elif exp_name == "CoinGame":
        def run(env):
            from lola.train_cg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  bs_mul=bs_mul,
                  gamma=gamma,
                  grid_size=grid_size,
                  lr=lr,
                  corrections=lola,
                  opp_model=opp_model,
                  hidden=hidden,
                  mem_efficient=mem_efficient)
    elif exp_name == "escape-room":
        def run(env, logdir):
            if n_agents == 2:
                from lola.train_er import train
            elif n_agents == 3:
                from lola.train_er_3player import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  gamma=gamma,
                  set_zero=0,
                  lr=lr,
                  corrections=lola,
                  simple_net=simple_net,
                  hidden1=64,
                  hidden2=32,
                  mem_efficient=mem_efficient,
                  logdir=logdir)
    elif exp_name == "asym-escape-room":
        def run(env, logdir):
            from lola.train_er_asym import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  gamma=gamma,
                  set_zero=0,
                  lr=lr,
                  simple_net=simple_net,
                  hidden1=64,
                  hidden2=32,
                  mem_efficient=mem_efficient,
                  logdir=logdir)            

    # Instantiate the environment
    if exp_name == "IPD":
        env = IPD(trace_length)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "IMP":
        env = IMP(trace_length)
        gamma = 0.9 if gamma is None else gamma
    elif exp_name == "CoinGame":
        env = CG(trace_length, batch_size, grid_size)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "escape-room":
        env = EscapeRoom(trace_length, n_agents)
        gamma = 0.99 if gamma is None else gamma
    elif exp_name == "asym-escape-room":
        env = EscapeRoomAsym(trace_length)
        gamma = 0.99 if gamma is None else gamma

    # Run training
    # for seed in range(trials):
    for seed in range(10, 20):
        # logdir = 'logs/{}/seed-{}'.format(exp_name, seed)
        # logdir = 'logs/{}/lr0p1-{}'.format(exp_name, seed)
        logdir = 'logs/{}/n{}-lr1-{}'.format(exp_name, n_agents, seed)
        logger.configure(dir=logdir)
        start_time = time.time()
        run(env, logdir)
        end_time  = time.time()
        logger.reset()

if __name__ == '__main__':
    main()
