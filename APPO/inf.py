import os
import time
import gym
# import pybullet_envs
import numpy as np
import random
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from rex_gym.envs.rex_gym_env import RexGymEnv
import sys
sys.path.append("..")
from rex_REDQ.train import ModHalfCheetahEnv, ModAntEnv, ModRexEnv, ModSpotEnv
import argparse
DEV = "cpu"


from train import ModHalfCheetahEnv, Agent_net
# ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurAlternatingLegsEnv-v0"

# from DPG_trot import Agent_net
# ENV_NAME = "MinitaurTrottingEnv-v0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-r", "--repeat_steps", help="Number of times to repeat each action", default=1, type=int)
    parser.add_argument("-d", "--hidden_dim", help="Hidden dim for both actor and critic", default=64, type=int)
    parser.add_argument("-e", "--env", help="which environment to use.",
                        choices=["rex", "ant", "halfcheetah", "spot"], required=True, )
    # Read arguments from command line
    args = parser.parse_args()

    exp_name = f"PPO_{args.env}"
    lr = None
    hidden_dim = None
    hidden_dim = args.hidden_dim
    repeat_steps = args.repeat_steps
    if args.env == "rex":
        env_fn = lambda render=False: ModRexEnv(hard_reset=False, render=render, terrain_id="plane")
    elif args.env == "ant":
        env_fn = lambda render=False: ModAntEnv(render)
    elif args.env == "halfcheetah":
        env_fn = lambda render=False: ModHalfCheetahEnv(render)
    elif args.env == "spot":
        env_fn = lambda render=False: ModSpotEnv(hard_reset=False, render=render)
    

    print("###############################################")
    print(f"Experiment Name: {exp_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Using env : {args.env}")
    print(f"Repeat steps : {args.repeat_steps}")
    print("###############################################")

    env = env_fn(render=True)
    agent = Agent_net(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_dim,
        dev=DEV
    ).to(DEV)
    state_dict = torch.load(f'model_{args.env}/model_{hidden_dim}.pt', map_location=DEV)
    agent.load_state_dict(state_dict=state_dict['actor_net'])
    # env.rex._pybullet_client.resetBasePositionAndOrientation(
        # env.rex.quadruped, [0.5, 0, 0.21], [0, 0, 0, 1])
    # control = [0]
    num_steps = 0
    with torch.no_grad():
        agent.logstd.zero_()
        for _ in range(1):
            is_done = False
            s = env.reset()
            env.render()
            while not is_done:
                a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV), noise=False)
                a = a.squeeze().cpu().numpy()
                # a = np.random.randn(env.action_space.shape[0]) / 1
                # a[0::3] = -0.5
                # print(a)
                for i in range(args.repeat_steps):
                    if not is_done:
                        s, r, is_done, _ = env.step(a) 
                        env.render()
                        num_steps += 1
                if num_steps > 1000:
                    print("done 1000 steps")
                # time.sleep(0.05)
            print("resetting")

        input("Press any key to exit\n")
        env.close()
