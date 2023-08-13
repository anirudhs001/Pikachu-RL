# %%

import os
from statistics import mean
import gym
import sys
sys.path.append("..")
from rex_REDQ.train import ModHalfCheetahEnv, ModAntEnv, ModRexEnv
from spotmicro.spotmicro.spot_gym_env import spotGymEnv
# from MiniCheetahEnv.gymMiniCheetahEnv.gym_MiniCheetahEnv.envs.mini_cheetah_env import MiniCheetahEnv
import pybullet
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
np.random.seed(43)
torch.cuda.manual_seed(43)
torch.manual_seed(43)
random.seed(43)

torch.backends.cudnn.deterministic=True

# %% [markdown]
# ##### ARGS

# %%
# ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurTrottingEnv-v0"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEV)
GAMMA = 0.99
GAE_LAMBDA = 0.98 
ENERGY_WEIGHT = 0
HEIGHT = 0.15
TEST_ITERS = 1000

NUM_ENVS_PER_PROC = 1 # TODO: somehow use multiple envs per proc
PROC_COUNT = 1 # TODO: somehow use multiple proces.
BATCH_SIZE = 256
TRAJECTORY_SIZE = 1024 * 2 + 1
PPO_EPOCHS = 20
PPO_EPS = 1e-3
MAX_EP_LEN = 1000

MAX_NOISE_EPSILON = 0.4
MIN_NOISE_EPSILON = 0.1
STEPS_TO_REDUCE_NOISE = 10_000



# %%
class ResLin(nn.Module):
    def __init__(self, in_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_n, in_n),
            nn.Tanh(),
            nn.Linear(in_n, in_n),
        )
        self.non_lin = nn.Tanh()
    def forward(self, x):
        x = x + self.net(x)
        x = self.non_lin(x)
        return x

class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int, hidden_dim: int, dev="cuda") -> None:
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
            nn.Tanh()
        )
        self.dev = dev
        self.logstd = nn.Parameter(torch.zeros(n_act, device=self.dev), requires_grad=True)
        self.noise_eps = 1


    def forward(self, x, noise=True):
        '''
        x : observation
        control : control value 
        '''
        x = self.l1(x)
        x = x + self.l2(x)
        mean = self.l3(x)
        if noise:
            a = mean + self.noise_eps * torch.randn(mean.shape)
            # std = torch.exp(self.logstd)
            # probs = Normal(mean, std)
            # a = probs.rsample()
        else:
            a = mean
        # mask = torch.ones(a.shape, device=self.dev)
        # mask[:, 0::3] = 0
        # out = a * mask
        return a


class Critic_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''
    def __init__(self, n_obs: int, hidden_dim:int, dev=DEV) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU() 
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        x = self.l1(obs)
        x = x + self.l2(x)
        x = self.l3(x)
        return x



# %% [markdown]
# #### Experience source

# %%

class Agent():
    def __init__(self, n_obs, n_act, lr_agent, lr_critic, hidden_dim) -> None:
        ## REPLAY BUFFER
        ## NETWORKS
        self.actor_net = Agent_net(n_obs, n_act, hidden_dim, DEV).to(DEV)
        self.critic_net = Critic_net(n_obs, hidden_dim, DEV).to(DEV)
        self.actor_optim = optim.Adam(
            self.actor_net.parameters(), lr=lr_agent)
        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), lr=lr_critic)
        self.hidden_dim = hidden_dim

    def save_nets(self, path="."):
        model_state = {
            "hidden_dim": self.hidden_dim,
            "actor_net": self.actor_net.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }
        torch.save(model_state, f"{path}/model_{self.hidden_dim}.pt")


    def load_nets(self, fname):
        model_state = torch.load(fname, map_location=DEV)
        self.actor_net.load_state_dict(model_state['actor_net'])
        self.critic_net.load_state_dict(model_state['critic_net'])
        self.actor_optim.load_state_dict(model_state['actor_optim'])
        self.critic_optim.load_state_dict(model_state['critic_optim'])
        return model_state["epoch"]



class Experience_source():
    def __init__(self, env_fn, repeat_steps, render=False):
        self.envs = [env_fn(render) for _ in range(NUM_ENVS_PER_PROC)]
        self.ep_rewards = [0 for _ in range(NUM_ENVS_PER_PROC)]
        self.curr_states = [env.reset() for env in self.envs]
        self.ep_lens = [0 for _ in range(NUM_ENVS_PER_PROC)]
        self.repeat_steps = repeat_steps

    @torch.no_grad()
    def step_env(self, actor_net, control, tg_height):
        control_one_hot = np.zeros(10)
        control_one_hot[control] = 1.
        total_rewards = []
        ep_lens = []
        exps = []
        for i, (s, env) in enumerate(zip(self.curr_states, self.envs)):
            r = 0
            s_ = np.append(s, values = [*control_one_hot, tg_height])
            s_v = torch.Tensor(s_).float().unsqueeze(0).to(DEV)
            a = actor_net(s_v, [control]).squeeze().cpu().numpy() 
            a = np.clip(a, -1, 1) # clipping the action to be in valid range, but will use the original unclipped action for training
            for _ in range(self.repeat_steps):
                new_s, r_, is_done, _ = env.step(a, control, tg_height)
                r += r_
                self.ep_lens[i] += 1
                if is_done or self.ep_lens[i] >= MAX_EP_LEN:
                    # is_done = True
                    break
            exps.append((s_, new_s, a, r, is_done))
            self.ep_rewards[i] += r
            if is_done or self.ep_lens[i] >= MAX_EP_LEN:
                new_s = env.reset()
                ep_lens.append(self.ep_lens[i])
                total_rewards.append(self.ep_rewards[i])
                self.ep_lens[i] = 0
                self.ep_rewards[i] = 0.
            self.curr_states[i] = new_s

        return total_rewards, ep_lens, exps


@torch.no_grad()
def calc_adv_n_ref(critic_net, states_v, trajectory):
    values_v = critic_net(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (_, _, _, r, is_done) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if is_done:
            delta = r - val
            last_gae = delta
        else:
            delta = r + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(DEV)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(DEV)
    return adv_v, ref_v


def calc_log_prob(mu_v, logstd_v, actions_v):
    p1 = -((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


# %%
def test_net(net:Agent_net, exp_src:Experience_source, count=1, device=DEV, random_control=False, control=0):
    rewards = 0.0
    steps = 0
    num_count = 4 if random_control else 1
    # tgt_heights = np.linspace(start = 0.16, stop=0.22, num=4)
    tgt_heights = [0.22]
    for tgt_height in tgt_heights:
        for i in range(num_count):
            if not random_control:
                c = control
            else:
                c = i
            while True:
                tot_r, _, _ = exp_src.step_env(net, c, tgt_height)
                steps += 1
                if len(tot_r) > 0 :
                    break
            rewards += tot_r[0]
    return rewards / count, steps / count


if __name__ == "__main__":
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
    parser.add_argument("--render", help="Render while training", action='store_true')
    parser.add_argument("--control", help="Train with Random Control and Target Height", action='store_true')
    parser.add_argument("-d", "--hidden_dim", help="Hidden dim for both actor and critic", default=64, type=int)
    parser.add_argument("-la", "--lr_agent", help="Learning rate for agent", default=1e-4, type=float)
    parser.add_argument("-lc", "--lr_critic", help="Learning rate for critic", default=3e-4, type=float)
    parser.add_argument("-e", "--env", help="which environment to use.",
                        choices=["rex", "ant", "halfcheetah", "spot"], required=True, )
    # Read arguments from command line
    args = parser.parse_args()

    save_path = None
    exp_name = f"PPO_{args.env}"
    lr = None
    hidden_dim = None
    save_path = f"{args.save_path}_{args.env}"
    hidden_dim = args.hidden_dim
    repeat_steps = args.repeat_steps
    render = False
    random_control=False
    control_shape = 11
    control = 0
    if args.render:
        render=True
    if args.control:
        random_control = True
    if args.comment:
        save_path = f"{save_path}_{args.comment}"
        exp_name = f"{args.comment}"
        if args.comment == "forward":
            control = 0
        if args.comment == "left":
            control = 1
        if args.comment == "back":
            control = 2
        if args.comment == "right":
            control = 3
        if args.comment == "sprint":
            control = 4
    if args.env == "rex":
        env_fn = lambda render=render: ModRexEnv(hard_reset=False, render=render, terrain_id="plane")
    elif args.env == "ant":
        env_fn = lambda render=render: ModAntEnv(render)
    elif args.env == "halfcheetah":
        env_fn = lambda render=render: ModHalfCheetahEnv(render)
    elif args.env == "spot":
        env_fn = lambda render=render: spotGymEnv(hard_reset=False, render=render)
    


    os.makedirs(save_path, exist_ok=True)
    print("###############################################")
    print(f"Saving checkpoints in: {save_path}")
    print(f"Experiment Name: {exp_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Using Learning Rates:: Agent: {args.lr_agent } Critic: {args.lr_critic}")
    print(f"Using env : {args.env}")
    print(f"Repeat steps : {args.repeat_steps}")
    print(f"Control : {control}")
    print("###############################################")

    test_env = env_fn(render=False)
    agent = Agent(
        test_env.observation_space.shape[0] + control_shape,
        test_env.action_space.shape[0],
        args.lr_agent,
        args.lr_critic,
        hidden_dim,
    )
    del test_env
    test_exp_src = Experience_source(env_fn, repeat_steps, render=False)
    exp_src = Experience_source(env_fn, repeat_steps, render=render)
    ep_rewards = collections.deque(maxlen=10)
    writer = SummaryWriter(comment=exp_name)

    r_mean = 0.
    best_r_mean = None
    trajectory = []
    step_idx = 0
    print("Starting Training")


    def generator():
        while True:
            yield


    pbar = tqdm(generator())
    last_ep_lens_for_each_control = [1, 1, 1, 1, 1]
    num_eps = 0
    last_ep_len = 1
    last_ep_r = 0.
    tg_height = 0.22
    for _ in pbar:
        step_idx += 1

        agent.actor_net.noise_eps = MAX_NOISE_EPSILON - (MAX_NOISE_EPSILON - MIN_NOISE_EPSILON) * (step_idx / STEPS_TO_REDUCE_NOISE)
        agent.actor_net.noise_eps = max(agent.actor_net.noise_eps, MIN_NOISE_EPSILON)
        rewards, ep_lens, steps = exp_src.step_env(actor_net=agent.actor_net, control=control, tg_height=tg_height)
        if len(rewards) > 0:
            last_ep_r = rewards[-1]
            last_ep_len = ep_lens[-1]
            num_eps += 1
            if num_eps % 10 == 0 and random_control:
                last_ep_lens_for_each_control[control] = last_ep_len
                # control = np.random.randint(low=0, high=4)
                # control = num_eps % 4
                ## pick the next control value depending how long episodes for each control are. 
                ## Eg. control=0 has short experiments compared to all other controls -> pick control=0 with higher
                ## probability so network gets more experience for it
                ps = [1/l for l in last_ep_lens_for_each_control]
                ps = [p * 1/sum(ps) for p in ps]
                ps[-1] = 1 - sum(ps[:-1])
                control = np.random.choice(np.arange(0, 5), p=ps)
                # tg_height = np.random.random() * 0.6 + 0.16
        

        trajectory += steps
        if step_idx % 100 == 0:
            writer.add_scalar("epsiode_reward", last_ep_r, step_idx)

        if step_idx % TEST_ITERS == 0:
            # checkpoint if best_r beaten
            test_r, test_ep_len = test_net(
                agent.actor_net, test_exp_src, device=DEV, random_control=random_control, control=control)
            writer.add_scalar("test_reward", test_r, step_idx)
            writer.add_scalar("test_steps", test_ep_len, step_idx)
            if best_r_mean is None or test_r >= best_r_mean:
                best_r_mean = test_r
                print(f"best_r updated: {best_r_mean : 0.4f}. Saving checkpoint")
                agent.save_nets(path=save_path)
            pbar.set_description(
                f"Last Test reward = {test_r:0.4f}, Best Test reward {best_r_mean:0.4f}")

        if len(trajectory) < TRAJECTORY_SIZE:
            continue

        # t is 5 tuple : s, new_s, a, r, is_done
        traj_states = [t[0] for t in trajectory]
        traj_actions = [t[2] for t in trajectory]
        traj_states_v = torch.FloatTensor(traj_states).to(DEV)
        traj_actions_v = torch.FloatTensor(traj_actions).to(DEV)

        traj_adv_v, traj_ref_v = calc_adv_n_ref(
            agent.critic_net, traj_states_v, trajectory)

        mu_v = agent.actor_net(traj_states_v)
        old_logprob_v = calc_log_prob(mu_v, agent.actor_net.logstd, traj_actions_v)
        # logstd_v = 0s because current actor only returns a single action value with 0 variance, and not a probability distribution

        # normalize advantages
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
        old_logprob_v = old_logprob_v[:-1].detach()

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        trajectory = trajectory[:-1]

        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0
        for epoch in range(PPO_EPOCHS):
            for batch_ofs in range(0, len(trajectory), BATCH_SIZE):
                states_v = traj_states_v[batch_ofs:batch_ofs + BATCH_SIZE]
                actions_v = traj_actions_v[batch_ofs:batch_ofs + BATCH_SIZE]
                batch_adv_v = traj_adv_v[batch_ofs:batch_ofs +
                                        BATCH_SIZE].unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_ofs + BATCH_SIZE]
                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + BATCH_SIZE]

                # critic training
                agent.critic_optim.zero_grad()
                value_v = agent.critic_net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                agent.critic_optim.step()

                # actor training
                agent.actor_optim.zero_grad()
                mu_v = agent.actor_net(states_v)
                logprob_pi_v = calc_log_prob(
                    mu_v, agent.actor_net.logstd, actions_v)
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                clipped_surr_v = batch_adv_v * \
                    torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                agent.actor_optim.step()

                sum_loss_value += loss_value_v.item()
                sum_loss_policy += loss_policy_v.item()
                count_steps += 1

        trajectory.clear()
        writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
        writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
        writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)
