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
from rex_REDQ.train import ModHalfCheetahEnv, ModRexEnv
from envs import ModAntEnv
from spotmicro.spotmicro.spot_gym_env import spotGymEnv
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
    parser.add_argument(
        "-c",
        "--comment",
        help="Optional comment to differentiate experiments. This will be added to both save_path and run name in tensorboard",
        default=""
    )
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
    obs_shape = 3
    if args.env == "rex":
        env_fn = lambda render=False: ModRexEnv(hard_reset=False, render=render, terrain_id="plane")
    elif args.env == "ant":
        env_fn = lambda render=False: ModAntEnv(render)
    elif args.env == "halfcheetah":
        env_fn = lambda render=False: ModHalfCheetahEnv(render)
    elif args.env == "spot":
        env_fn = lambda render=False: spotGymEnv(hard_reset=False, render=render)
    

    print("###############################################")
    print(f"Experiment Name: {exp_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Using env : {args.env}")
    print(f"Repeat steps : {args.repeat_steps}")
    print("###############################################")

    env = env_fn(render=True)
    agent = Agent_net(
        env.observation_space.shape[0] + obs_shape,
        env.action_space.shape[0],
        hidden_dim,
        dev=DEV
    ).to(DEV)
    load_path = f'model_{args.env}'
    if args.comment:
        load_path = f'{load_path}_{args.comment}'
    state_dict = torch.load(f'{load_path}/model_{hidden_dim}.pt', map_location=DEV)
    agent.load_state_dict(state_dict=state_dict['actor_net'])
    # env.rex._pybullet_client.resetBasePositionAndOrientation(
        # env.rex.quadruped, [0.5, 0, 0.21], [0, 0, 0, 1])
    # control = [0]
    num_steps = 0
    with torch.no_grad():
        agent.logstd.zero_()
        controls = np.linspace(-0.20, 0.20, 3)
        for c in controls:
            print(c)
            is_done = False
            s = env.reset()
            env.render()
            num_steps = 0
            while not is_done:
                s = np.append(s, [1, 0, c])
                a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV), noise=False)
                a = a.squeeze().cpu().numpy()
                # a = np.random.randn(env.action_space.shape[0]) / 1
                # a[0::3] = -0.5
                # print(a)
                if not is_done:
                    s, r, _, _, is_done = env.step(a, args.repeat_steps) 
                    env.render()
                    num_steps += args.repeat_steps
                if num_steps > 500:
                    print("done 500 steps")
                    break
                time.sleep(0.01)
            print("resetting")

        input("Press any key to exit\n")
        env.close()
