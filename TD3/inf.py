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
import sys
from torch.distributions.normal import Normal
import sys
sys.path.append("..")
from rex_REDQ.train import ModHalfCheetahEnv, ModAntEnv, ModRexEnv, ModSpotEnv
from train import Actor_critic, Agent_net, Q_net
import time
import argparse
DEV = "cpu"

# from rex_gym.envs import rex_gym_env

# from rex import RexGymEnvMod
# from rex import MAX_EP_LEN

# from MiniCheetahEnv.gymMiniCheetahEnv.gym_MiniCheetahEnv.envs.mini_cheetah_env import MiniCheetahEnv
# ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurAlternatingLegsEnv-v0"

# ENV_NAME = "MinitaurTrottingEnv-v0"
# ENV_NAME="HalfCheetah"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--save_path",
                        help="Path to save checkpoints", default="model")
    parser.add_argument("-r", "--repeat_steps", help="Number of times to repeat each action", default=1, type=int)
    parser.add_argument("-d", "--hidden_dim", help="Hidden dim for both actor and critic", default=64, type=int)
    parser.add_argument("-e", "--env", help="which environment to use.",
                        choices=["rex", "ant", "halfcheetah", "spot"], required=True, )
    # Read arguments from command line
    args = parser.parse_args()

    save_path = f"{args.save_path}_{args.env}"
    hidden_dim = args.hidden_dim
    repeat_steps = args.repeat_steps
    if args.env == "rex":
        env_fn = lambda render: ModRexEnv(hard_reset=False, render=render, terrain_id="plane")
    elif args.env == "ant":
        env_fn = lambda render: ModAntEnv(render)
    elif args.env == "halfcheetah":
        env_fn = lambda render: ModHalfCheetahEnv()
    elif args.env == "spot":
        env_fn = lambda render: ModSpotEnv(hard_reset=False, render=render)
    
    env = env_fn(render=True)
    # print(env.action_space, env.action_space.shape)
    # ac = torch.load(f'{save_path}/pyt_save/model{args.hidden_dim}.pt', map_location=DEV)
    # agent = ac.pi
    # del ac
    # env.rex._pybullet_client.resetBasePositionAndOrientation(
        # env.rex.quadruped, [0.5, 0, 0.21], [0, 0, 0, 1])
    # control = [0]
    num_steps = 0
    with torch.no_grad():
        for _ in range(1):
            is_done = False
            s = env.reset()
            while not is_done:
                # a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV))
                # a = a.squeeze().cpu().numpy()
                a = np.random.randn(env.action_space.shape[0]) / 10
                # print(a)
                for i in range(args.repeat_steps):
                    if not is_done:
                        s, r, is_done, _ = env.step(a) 
                        env.render()
                        num_steps += 1
                # if num_steps > 1000:
                    # print("done 1000 steps")
                time.sleep(0.1)
            print("resetting")

        input("Press any key to exit\n")
        env.close()
