# %%
# import os
from statistics import mean
# import time
# import gym
# import pybullet_envs
# import pybullet
# import argparse
from tensorboardX import SummaryWriter
from queue import Queue
import numpy as np
# import random
from tqdm import tqdm
# import collections
import multiprocessing as mp
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from spinup import sac_pytorch
# from rex_gym.envs import rex_gym_env
import sys
sys.path.append("..")
from spotmicro.spotmicro.spot_gym_env import spotGymEnv
import argparse
np.random.seed(43)
torch.manual_seed(43)



BATCH_SIZE = 100
LR = 1e-4
BUF_SIZE = 500_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10_000
# print(DEV)
REPEAT_STEPS = 1
MAX_EP_LEN = 2_000
GAMMA = 0.99
ALPHA = 0.2

ENERGY_WEIGHT = 0
HEIGHT = 1.2


class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh() 
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mu = nn.Sequential(
            nn.Linear(64, n_act),
        )
        self.log_std = nn.Sequential(
            nn.Linear(64, n_act),
        )
        self.dev = DEV

    def forward(self, x):
        '''
        x : observation
        '''
        # x = x.to(self.dev)
        x = self.l1(x)
        x = x + self.l2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_pi_a = (dist.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u)))).sum()
        # source : https://github.com/XinJingHao/SAC-Continuous-Pytorch/blob/07b6a5d0cb70f11db793ae39b1024a17379b5534/SAC.py#L51
        return a, log_pi_a


class Q_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''
    def __init__(self, n_obs: int, n_act: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs+n_act, 64),
            nn.ReLU() 
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.dev = DEV
    def forward(self, obs, act):
        # obs = obs.to(self.dev)
        # act = act.to(self.dev)
        x = torch.cat([obs, act], dim=1)
        q = self.l1(x)
        q = q + self.l2(q)
        q = self.l3(q)
        return q



def env_fn():
    return spotGymEnv(hard_reset=False, render=False)


class Actor_critic(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        n_obs = obs_space.shape[0]
        n_act = act_space.shape[0]
        self.pi = Agent_net(n_obs, n_act).to(DEV)
        self.q1 = Q_net(n_obs, n_act).to(DEV)
        self.q2 = Q_net(n_obs, n_act).to(DEV)

    def act(self, obs, *args):
        a, _ = self.pi(obs)
        a = a.detach().cpu().numpy()
        return a


if __name__ == "__main__":
    print("TRAINING")
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-s", "--save_path", help = "Path to save checkpoints", default=".")
    parser.add_argument(
        "-c",
        "--comment",
        help="Optional comment to differentiate experiments. This will be added to both save_path and run name in tensorboard",
        default=""
    )
    parser.add_argument("-l", "--lr", help = "Learning rate", default=LR)
    
    # Read arguments from command line
    args = parser.parse_args()
    
    save_path = "./"
    exp_name = "spot_sac"
    lr = LR
    if args.save_path:
        save_path = args.save_path
    if args.comment:
        save_path += f"_{args.comment}"
        exp_name += f"_{args.comment}"
    if args.lr:
        lr = float(args.lr)

    print(f"Saving checkpoints in: {save_path}") 
    print(f"Experiment Name: {exp_name}")
    print(f"Using Learning Rate: {lr}")

    sac_pytorch(env_fn,
                Actor_critic,
                logger_kwargs={
                    "output_dir": save_path, "exp_name": exp_name},
                lr=lr,
                num_test_episodes=2,
                batch_size=BATCH_SIZE,
                replay_size=BUF_SIZE,
                epochs=EPOCHS
                )
    
