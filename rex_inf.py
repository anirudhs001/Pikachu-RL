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

from rex_gym.envs import rex_gym_env
from rex_networks import Agent_net
from rex import RexGymEnvMod
from rex import MAX_EP_LEN

# ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurAlternatingLegsEnv-v0"

# from DPG_trot import Agent_net
# ENV_NAME = "MinitaurTrottingEnv-v0"



if __name__ == "__main__":
    print(sys.argv)
    control = int(sys.argv[1])
    print("\n############\nCONTROL : ", control, "\n############")
    # spec = gym.envs.registry.spec(ENV_NAME)
    # spec.kwargs['render'] = True
    # env = gym.make(ENV_NAME)
    # env = rex_gym_env.RexGymEnv(render=True, terrain_id="plane")
    env = RexGymEnvMod(render=True, terrain_id="plane")

    pos = env.rex.GetBasePosition()
    print(pos)
    agent = Agent_net(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        dev=DEV
    ).to(DEV)
    # print(env.observation_space, env.observation_space.shape)
    # print(env.action_space, env.action_space.shape)
    state_dict = torch.load(f'rex_models/model_{control}.pt', map_location=DEV)
    agent.load_state_dict(state_dict=state_dict['action_model'])
    # env.rex._pybullet_client.resetBasePositionAndOrientation(
        # env.rex.quadruped, [0.5, 0, 0.21], [0, 0, 0, 1])
    # control = [0]
    num_steps = 0
    with torch.no_grad():
        for _ in range(1):
            is_done = False
            s = env.reset()
            while not is_done:
                a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV))
                a = a.squeeze().cpu().numpy()
                # a = np.random.randn(env.action_space.shape[0]) / 1
                # a[0::3] = -0.5
                # print(a)
                for i in range(5):
                    if not is_done:
                        s, r, is_done, _ = env.step(a) 
                        num_steps += 1
                if num_steps > 1000:
                    print("done 1000 steps")
                #time.sleep(0.2)
            print("resetting")

        input("Press any key to exit\n")
        env.close()
