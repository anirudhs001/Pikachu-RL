import os
import time
from tkinter import W
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
import sys
DEV = "cpu"

# from rex_gym.envs import rex_gym_env
from DPG import Agent_net
ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurAlternatingLegsEnv-v0"

# from DPG_trot import Agent_net
# ENV_NAME = "MinitaurTrottingEnv-v0"



if __name__ == "__main__":
    print(sys.argv)
    control = int(sys.argv[1])
    print("\n############\nCONTROL : ", control, "\n############")
    spec = gym.envs.registry.spec(ENV_NAME)
    spec.kwargs['render'] = True
    env = gym.make(ENV_NAME)
    # env = rex_gym_env.RexGymEnv(render=True, terrain_id="plane")

    pos = env.minitaur.GetBasePosition()
    print(pos)
    agent = Agent_net(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        dev=DEV
    ).to(DEV)
    # print(env.observation_space, env.observation_space.shape)
    # print(env.action_space, env.action_space.shape)
    state_dict = torch.load(f'minitaur_models/model_{control}.pt', map_location=DEV)
    agent.load_state_dict(state_dict=state_dict['action_model'])
    is_done = False
    s = env.reset()
    # env.minitaur._pybullet_client.resetBasePositionAndOrientation(
        # env.minitaur.quadruped, [0, 0.5, 0.2], [0, 0, 0, 1])
    control = [2]
    with torch.no_grad():
        while not is_done:
            a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV))
            a = a.squeeze().cpu().numpy()
            # a = np.random.randn(env.action_space.shape[0]) / 10
            # a[0::3] = 0
            # print(a)
            for i in range(5):
                if not is_done:
                    s, r, is_done, _ = env.step(a) 
            # time.sleep(0.1)
        input("Press any key to exit\n")
        env.close()
