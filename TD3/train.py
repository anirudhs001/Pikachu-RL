# %%
import os
from statistics import mean
import time
# import time
import gym
# import pybullet_envs
# import pybullet
# import argparse
from tensorboardX import SummaryWriter
import sys
sys.path.append("..")
from rex_REDQ.train import ModHalfCheetahEnv, ModAntEnv, ModRexEnv, spotGymEnv
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
# from spinup import sac_pytorch
from spinup import td3_pytorch
from rex_gym.envs import rex_gym_env
import argparse
np.random.seed(43)
torch.manual_seed(43)



BATCH_SIZE = 100
BUF_SIZE = 500_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10_000
# print(DEV)
MAX_EP_LEN = 1_000
GAMMA = 0.99
ALPHA = 0.2



class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int, hidden_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.Tanh() 
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_dim, n_act),
        )
        self.dev = DEV

    def forward(self, x, noise=True):
        '''
        x : observation
        '''
        # x = x.to(self.dev)
        x = self.l1(x)
        x = x + self.l2(x)
        a = self.l3(x)
        return a


class Q_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''
    def __init__(self, n_obs: int, n_act: int, hidden_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs+n_act, hidden_dim),
            nn.ReLU() 
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
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

class Actor_critic(nn.Module):
    def __init__(self, obs_space, act_space, hidden_dim):
        super().__init__()
        n_obs = obs_space.shape[0]
        n_act = act_space.shape[0]
        self.pi = Agent_net(n_obs, n_act, hidden_dim).to(DEV)
        self.q1 = Q_net(n_obs, n_act, hidden_dim).to(DEV)
        self.q2 = Q_net(n_obs, n_act, hidden_dim).to(DEV)

    def act(self, obs, *args):
        a = self.pi(obs)
        a = a.detach().cpu().numpy()
        return a


if __name__ == "__main__":
    print("TRAINING")
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--save_path",
                        help="Path to save checkpoints", default="model")
    parser.add_argument(
        "-c",
        "--comment",
        help="Optional comment to differentiate experiments. This will be added to both save_path and run name in tensorboard",
        default=""
    )
    parser.add_argument("-r", "--repeat_steps", help="Number of times to repeat each action", default=1, type=int)
    parser.add_argument("-d", "--hidden_dim", help="Hidden dim for both actor and critic", default=64, type=int)
    parser.add_argument("-la", "--lr_agent", help="Learning rate for agent", default=1e-4, type=float)
    parser.add_argument("-lc", "--lr_critic", help="Learning rate for critic", default=3e-4, type=float)
    parser.add_argument("-e", "--env", help="which environment to use.",
                        choices=["rex", "ant", "halfcheetah", "spot"], required=True, )
    # Read arguments from command line
    args = parser.parse_args()

    save_path = None
    exp_name = f"TD3_{args.env}"
    lr = None
    hidden_dim = None
    save_path = f"{args.save_path}_{args.env}"
    hidden_dim = args.hidden_dim
    repeat_steps = args.repeat_steps
    if args.comment:
        save_path = f"{save_path}_{args.comment}"
        exp_name = f"{args.comment}"
    if args.env == "rex":
        env_fn = lambda : ModRexEnv(hard_reset=False, render=False, terrain_id="plane")
    elif args.env == "ant":
        env_fn = lambda : ModAntEnv()
    elif args.env == "halfcheetah":
        env_fn = lambda : ModHalfCheetahEnv()
    elif args.env == "spot":
        env_fn = lambda : spotGymEnv(hard_reset=False, render=False)
    


    os.makedirs(save_path, exist_ok=True)
    print("###############################################")
    print(f"Saving checkpoints in: {save_path}")
    print(f"Experiment Name: {exp_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Using Learning Rates:: Agent: {args.lr_agent } Critic: {args.lr_critic}")
    print(f"Using env : {args.env}")
    print(f"Repeat steps : {args.repeat_steps}")
    print("###############################################")

    input("Press Enter to continue...")

    td3_pytorch(env_fn,
                Actor_critic,
                logger_kwargs={
                    "output_dir": save_path, "exp_name": exp_name},
                ac_kwargs={'hidden_dim':hidden_dim},
                pi_lr=args.lr_agent,
                q_lr=args.lr_critic,
                num_test_episodes=2,
                model_suffix=hidden_dim,
                epochs=1000,
    )
    
