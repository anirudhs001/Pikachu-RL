from DPG import ENV_NAME, Agent
import os
import time
import gym
import pybullet_envs
import numpy as np
import random
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

ENV_NAME = "MinitaurBulletEnv-v0"
DEV = "cuda"

class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int, dev="cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 400),
            nn.LeakyReLU(0.1), 
            nn.Linear(400, 300),
            nn.LeakyReLU(0.1),
            nn.Linear(300, n_act),
            nn.Tanh()
        )
        self.dev = dev

    def forward(self, x):
        out = self.net(x)
        return out

if __name__ == "__main__":
    spec = gym.envs.registry.spec(ENV_NAME)
    spec.kwargs['render'] = True
    env = gym.make(ENV_NAME)
    agent = Agent_net(
        env.observation_space.shape[0],
        env.action_space.shape[0],
    ).to(DEV)
    state_dict = torch.load('model_0.pt')
    agent.load_state_dict(state_dict=state_dict['action_model'])

    is_done = False
    s = env.reset()
    with torch.no_grad():
        while not is_done:
            a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV))
            a = a.squeeze().cpu().numpy()
            for i in range(5):
                if not is_done:
                    s, r, is_done, _ = env.step(a) 
            # time.sleep(0.1)
        input("Press any key to exit\n")
        env.close()