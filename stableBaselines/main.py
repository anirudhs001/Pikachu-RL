import sys
sys.path.append("/Users/anirudhsingh/MISC/playground/Pikachu-RL")
import argparse
from tensorboardX import SummaryWriter
from queue import Queue
import numpy as np
import random
from tqdm import tqdm
import collections
import multiprocessing as mp
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import os
from statistics import mean
import gymnasium
import time
import termios
import tty
import asyncio
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC
from stable_baselines3.td3 import TD3
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
import warnings
from Envs.envs import ModAntEnv_V2, ModAntEnv
from typing import List

np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)
random.seed(42)


# Ignore all warnings
warnings.filterwarnings("ignore")


class Custom_callback(BaseCallback):
    def __init__(self):
        super(Custom_callback, self).__init__()
        self.controls = [
            ((1, 0, 0, 0, 0), 0),
            # ((-1, 0, 0, 0, 0), 0),
            # ((0, 1, 0, 0, 0), 1),
            # ((0, -1, 0, 0, 0), 1),
            # ((0, 0, 0, 0, 0), 2),
            # ((0, 0, 1, 0, 0), 2),
            # ((0, 0, 0, 1, 0), 3),
            # ((0, 0, 0, 0, 1), 4),
        ]
        self.control_interval = None

    def _on_step(self) -> bool:
        if self.control_interval is None:
            self.control_interval = self.model.env.num_envs * self.model.n_steps
            # self.model.policy.optimizer.zero_grad()
            return True

        if self.num_timesteps % self.control_interval == 0:
            # if self.num_timesteps // self.control_interval % len(self.controls) == 0:
            # self.model.policy.optimizer.step()
            # self.model.policy.optimizer.zero_grad()
            # self.model.rollout_buffer.reset() # no need to reset anything if we call train() after and only after this fn
            control, head_idx = self.controls[
                self.num_timesteps // self.control_interval % len(self.controls)
            ]
            self.model.policy.mlp_extractor.policy_net.head_idx = head_idx
            ret = self.model.env.env_method("set_control", control)
            # print("Control set to: ", control, "Return: ", ret)
        return True


class Multi_head_policy_net(nn.Module):
    def __init__(
        self,
        inp_size: int,
        pi: List[int] = [32, 32],
        num_heads: int = 5,
        activation_fn=nn.Tanh,
    ) -> None:
        super().__init__()
        pi.insert(0, inp_size)
        self.pi_heads = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(pi[i], pi[i + 1]) for i in range(len(pi) - 1)])
                for _ in range(num_heads)
            ]
        )
        self.activation_fn = activation_fn()
        self.head_idx = 0

    def forward(self, x: torch.Tensor, head_idx: int = None, **kwargs):
        if head_idx is not None:
            self.head_idx = head_idx

        for layer in self.pi_heads[self.head_idx]:
            x = self.activation_fn(layer(x))

        return x


class Log:
    def __init__(self, exp_name="ppo2_modAntEnv_V2"):
        self.step_idx = 0
        self.writer = SummaryWriter(comment=exp_name)

    def log(self, writer, name, value):
        self.step_idx += 1
        writer.add_scalar(name, value, self.step_idx)


async def read_keypress(queue):
    # Set the terminal in raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())

        loop = asyncio.get_event_loop()
        loop.add_reader(sys.stdin, enqueue_keypress, queue)

        while True:
            await asyncio.sleep(1)  # Keep the task running

    except KeyboardInterrupt:
        sys.stdout.flush()  # Ensure message is displayed
        print("\nGracefully handling Ctrl+C...")

    finally:
        # Restore the terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        loop.remove_reader(sys.stdin)


def enqueue_keypress(queue):
    key = sys.stdin.read(1)
    asyncio.ensure_future(queue.put(key))


async def run_simulation(queue, algo, device, comment, checkpoint):
    env = ModAntEnv_V2(render_mode="human")
    path = f"checkpoints_{algo}_{device}"
    path += "_" + comment if comment else ""
    path += f"/rl_model_{checkpoint}_steps.zip"
    if algo == "PPO":
        # policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
        # model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, device="cpu")
        # custom_policy_net = Multi_head_policy_net(model.policy.features_dim, pi=[32, 32], num_heads=5).to('cpu')
        # model.policy.mlp_extractor.policy_net = custom_policy_net
        # state_dict = load_from_zip_file(path, device="cpu")
        # model.policy.load_state_dict(state_dict[1]['policy'])
        model = PPO.load(path, device="cpu")
    elif algo == "SAC":
        model = SAC.load(path, device="cpu")
    control = (1, 0)
    obs = env.reset()
    print("Simulation running...")

    while True:
        # Check if there are any keypresses in the queue
        if not queue.empty():
            key = await queue.get()
            # Process the keypress in the simulation or perform any other desired action
            if key == "w":
                control = (1, 0)
            elif key == "s":
                control = (-1, 0)
            elif key == "a":
                control = (0, 1)
            elif key == "d":
                control = (0, -1)
            # elif key == "j":
            #     control = (0, 0, 0, 0, 0)
            #     head_idx = 2
            # elif key == "k":
            #     control = (0, 0, 1, 0, 0)
            #     head_idx = 2
            # elif key == "l":
            #     control = (0, 0, 0, 1, 0)
            #     head_idx = 3
            # elif key == ";":
            #     control = (0, 0, 0, 0, 1)
            #     head_idx = 4
            elif key == "q":
                exit()
            else:
                control = (0, 0, 0, 0, 0)
                head_idx = 2
            env.control = control
            print("env.control: ", env.control, "\n")
        # Run your simulation logic here
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action, repeat_steps=1)
        env.render()
        await asyncio.sleep(0.01)  # Simulating the passage of time


async def live_control(algo, device, comment, checkpoint):

    queue = asyncio.Queue()

    keypress_task = asyncio.create_task(read_keypress(queue))
    simulation_task = asyncio.create_task(
        run_simulation(queue, algo, device, comment, checkpoint)
    )
    try:
        await asyncio.gather(keypress_task, simulation_task)
        # await asyncio.gather()
    except KeyboardInterrupt:
        print("Simulation stopped.")


if __name__ == "__main__":

    # read train/inf arg from command line:
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--num_envs", type=int, default=4)
    parser.add_argument("-ns", "--steps", type=int, default=1_000_000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inf", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=int, default=0)
    parser.add_argument("-v", "--vel_control", type=float, default=1)
    parser.add_argument("-r", "--rot_control", type=int, default=0)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-a", "--algo", type=str, default="ppo")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-l", "--log_path", type=str, default=None)
    parser.add_argument("-maxe", "--max_episode_length", type=int, default=None)
    parser.add_argument("--comment", type=str, default="")
    # arg for learning rate

    args = parser.parse_args()
    train = args.train
    n_envs = args.num_envs
    n_steps = args.steps
    inf = args.inf
    checkpoint = args.checkpoint
    v = args.vel_control
    r = args.rot_control
    algo = args.algo
    lr = args.learning_rate
    device = args.device
    comment = args.comment
    log_path = args.log_path
    max_episode_length = args.max_episode_length
    remote = args.remote
    train = train & (not inf) & (not remote)
    inf = inf & (not train) & (not remote)
    remote = remote & (not train) & (not inf)
    # print all args
    print("====================================")
    print("train: ", train)
    print("inf: ", inf)
    print("remote: ", remote)
    print("n_envs: ", n_envs)
    print("n_steps: ", n_steps)
    print("checkpoint: ", checkpoint)
    print("vel_control: ", v)
    print("rot_control: ", r)
    print("algo: ", algo)
    print("learning_rate: ", lr)
    print("device: ", device)
    print("comment: ", comment)
    print("log_path: ", log_path)
    print("max_episode_length: ", max_episode_length)
    print("====================================")

    if log_path:
        os.makedirs(log_path, exist_ok=True)

    if train:
        save_path = f"./checkpoints_{algo}_{device}"
        save_path += f"_{comment}/" if comment else "/"
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint, save_path=save_path
        )
        callbacks = [checkpoint_callback]
        if algo == "PPO":
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=[dict(pi=[32, 32], vf=[32, 32])],
            )
            env_fn = lambda *args, **kwargs: ModAntEnv_V2(
                *args, max_ep_length=max_episode_length, **kwargs
            )
            env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            # policy = lambda *args, **kwargs: ActorCriticPolicy(
            # *args, env.observation_space, env.action_space, lambda x: 3e-4, net_arch=dict(pi=[2, 2], vf=[32, 32]), **kwargs)
            learner = lambda *args, **kwargs: PPO(
                *args,
                policy="MlpPolicy",
                policy_kwargs=policy_kwargs,
                env=env,
                n_steps=1024,
                batch_size=128,
                **kwargs,
            )
            # custom_callback = Custom_callback()
            # callbacks.append(custom_callback)
        elif algo == "SAC":
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh, net_arch=[128, 128], use_sde=True
            )
            learner = lambda *args, **kwargs: SAC(
                *args,
                policy="MlpPolicy",
                env=ModAntEnv_V2(),
                policy_kwargs=policy_kwargs,
                **kwargs,
            )
        elif algo == "TD3":
            learner = lambda *args, **kwargs: TD3(
                "MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1
            )
        else:
            raise NotImplementedError
        model = learner(
            verbose=1, learning_rate=lr, device=device, tensorboard_log=log_path
        )
        # change policy net with custom policy net
        # custom_policy_net = Multi_head_policy_net(model.policy.features_dim, pi=[32, 32], num_heads=5).to(device)
        # model.policy.mlp_extractor.policy_net = custom_policy_net
        model.learn(total_timesteps=n_steps, callback=callbacks)
    if inf:
        env = ModAntEnv_V2(render_mode="human")
        path = f"checkpoints_{algo}_{device}"
        path += "_" + comment if comment else ""
        path += f"/rl_model_{checkpoint}_steps.zip"
        if algo == "PPO":
            model = PPO.load(path, device="cpu")
        elif algo == "SAC":
            model = SAC.load(path, device="cpu")
        obs = env.reset(control=(v, r))

        while True:
            action, _states = model.predict(obs)
            print("obs: ", obs)
            obs, rewards, done, info = env.step(action, repeat_steps=1)
            print("rewards: ", rewards)
            env.render()
            print(done, action)
            time.sleep(0.01)
        print("done")
    if remote:
        # print("remote")
        asyncio.run(live_control(algo, device, comment, checkpoint))
