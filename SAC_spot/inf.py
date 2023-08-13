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
DEV = "cpu"

# from rex_gym.envs import rex_gym_env
from train import Actor_critic, Agent_net, Q_net
# from spotmicro.GymEnvs import spot_bezier_env
sys.path.append('..')

from spotmicro.spotmicro.spot_gym_env import spotGymEnv 
# from rex import RexGymEnvMod
# from rex import MAX_EP_LEN

# from MiniCheetahEnv.gymMiniCheetahEnv.gym_MiniCheetahEnv.envs.mini_cheetah_env import MiniCheetahEnv
# ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurAlternatingLegsEnv-v0"

# ENV_NAME = "MinitaurTrottingEnv-v0"
# ENV_NAME="HalfCheetah"


if __name__ == "__main__":
    print(sys.argv)
    control = int(sys.argv[1])
    print("\n############\nCONTROL : ", control, "\n############")
    # spec = gym.envs.registry.spec(ENV_NAME)
    # spec.kwargs['render'] = True

    # env = gym.make(ENV_NAME)
    
    # env = ModRexEnv(hard_reset=False, render=True, terrain_id="plane")
    env = spotGymEnv(render=True, hard_reset=False)
    # env = MiniCheetahEnv(render=True, on_rack=False)
    # env = RexGymEnvMod(render=True, terrain_id="plane")
    # pos = env.rex.GetBasePosition()
    # print(pos)
    ac = Actor_critic(env.observation_space, env.action_space)
    
    # print(env.observation_space, env.observation_space.shape)
    # print(env.action_space, env.action_space.shape)
    ac = torch.load(f'pyt_save/model.pt', map_location=DEV)
    agent = ac.pi
    del ac
    # env.rex._pybullet_client.resetBasePositionAndOrientation(
        # env.rex.quadruped, [0.5, 0, 0.21], [0, 0, 0, 1])
    # control = [0]
    num_steps = 0
    with torch.no_grad():
        for _ in range(1):
            is_done = False
            s = env.reset()
            while not is_done:
                a, _ = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEV))
                a = a.squeeze().cpu().numpy()
                # a = np.random.randn(env.action_space.shape[0]) 
                # a[0::3] = 0
                # print(a)
                for i in range(1):
                    if not is_done:
                        s, r, is_done, _ = env.step(a) 
                        env.render()
                        num_steps += 1
                if num_steps > 1000:
                    print("done 1000 steps")
                # time.sleep(0.2)
            print("resetting")

        input("Press any key to exit\n")
        env.close()
