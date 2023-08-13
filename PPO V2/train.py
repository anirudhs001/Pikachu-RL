import pdb
import math
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch
import multiprocessing as mp
import collections
from tqdm import tqdm
import random
import numpy as np
from queue import Queue
from tensorboardX import SummaryWriter
import argparse
# import pybullet
import sys
sys.path.append("..")
from spotmicro.spotmicro.spot_gym_env import spotGymEnv
from envs import ModAntEnv
from rex_REDQ.train import ModHalfCheetahEnv, ModRexEnv
import os
from statistics import mean
import gym
import sys
# from MiniCheetahEnv.gymMiniCheetahEnv.gym_MiniCheetahEnv.envs.mini_cheetah_env import MiniCheetahEnv
np.random.seed(43)
torch.cuda.manual_seed(43)
torch.manual_seed(43)
random.seed(43)

torch.backends.cudnn.deterministic = True

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

NUM_ENVS_PER_PROC = 1  # TODO: somehow use multiple envs per proc
PROC_COUNT = 1  # TODO: somehow use multiple proces.
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

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int, dev="cpu") -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.LeakyReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.l3 = nn.Sequential(
            nn.Linear(128, n_act),
            nn.LeakyReLU(),
        )
        self.dev = dev
        self.logstd = nn.Parameter(torch.zeros(
            n_act, device=self.dev), requires_grad=True)
        self.noise_eps = 1

    def forward(self, x, noise=True):
        '''
        x : observation
        control : control value 
        '''
        x = self.l1(x)
        x = self.l2(x)
        mean = self.l3(x)
        if noise:
            a = mean + self.noise_eps * \
                torch.randn(mean.shape, device=self.dev)
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

    def __init__(self, n_obs: int, hidden_dim: int, dev=DEV) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        x = self.l1(obs)
        x = self.l2(x)
        x = self.l3(x)
        return x


# %% [markdown]
# #### Experience source

# %%

class Agent():
    def __init__(self, n_obs, n_act, lr_agent, lr_critic, hidden_dim) -> None:
        # REPLAY BUFFER
        # NETWORKS
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
disable laptop monitdisable laptop monidisable lapdisable laptop monittop monitt
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
        self.ep_reward_comps = [0 for _ in range(NUM_ENVS_PER_PROC)]
        self.curr_states = [env.reset() for env in self.envs]
        self.ep_lens = [0 for _ in range(NUM_ENVS_PER_PROC)]
        self.repeat_steps = repeat_steps

    @torch.no_grad()
    def step_env(self, actor_net, controls, repeat_steps=1):
        total_rewards = []
        ep_lens = []
        exps = []
        for i, (s, c, env) in enumerate(zip(self.curr_states, controls, self.envs)):
            # state = bot's limb angles + desired motion direction + desired head direction + desired rotation
            s_ = np.append(s, values=c)
            s_v = torch.Tensor(s_).float().unsqueeze(0).to(DEV)
            a = actor_net(s_v).squeeze().cpu().numpy()
            # clipping the action to be in valid range, but will use the original unclipped action for training
            a_ = np.clip(a, -1, 1)
            new_s, r, r_comps, reached_targ, is_done = env.step(a_, repeat_steps)
            # if reached_targ:
            #     # calc new targ
            #     env.update_targ(c)
            #     # print(f'target updated:: controls: {c}. new targ: {env.targ}')
            self.ep_lens[i] += repeat_steps
            exps.append((s_, new_s, a, r, is_done))
            self.ep_rewards[i] += r
            self.ep_reward_comps[i] += r_comps
            if is_done or self.ep_lens[i] >= MAX_EP_LEN:
                new_s = env.reset()
                ep_lens.append(self.ep_lens[i])
                total_rewards.append((self.ep_rewards[i], self.ep_reward_comps[i]))
                self.ep_lens[i] = 0
                self.ep_rewards[i] = 0.
                self.ep_reward_comps[i] = 0.
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
def test_net(net: Agent_net, exp_src: Experience_source, count=1):
    rewards = 0.0
    steps = 0
    controls = [(1, 0, t) for t in np.linspace(-0.25, 0.25, 4)]
    for c in controls:
        while True:
            tot_r, _, _ = exp_src.step_env(net, [c])
            steps += 1
            if len(tot_r) > 0:
                break
        rewards += tot_r[-1][0]
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
    parser.add_argument("-r", "--repeat_steps",
                        help="Number of times to repeat each action", default=1, type=int)
    parser.add_argument(
        "--render", help="Render while training", action='store_true')
    parser.add_argument(
        "--random_control", help="Train with Random Control and Target Height", action='store_true')
    parser.add_argument(
        "-d", "--hidden_dim", help="Hidden dim for both actor and critic", default=64, type=int)
    parser.add_argument(
        "-la", "--lr_agent", help="Learning rate for agent", default=1e-4, type=float)
    parser.add_argument(
        "-lc", "--lr_critic", help="Learning rate for critic", default=3e-4, type=float)
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
    random_control = False
    # control are 2 floats for each env
    controls = [(1,0,0) for _ in range(NUM_ENVS_PER_PROC)]
    control_shape = 3  # two for control
    if args.render:
        render = True
    if args.random_control:
        random_control = True
    if args.env == "rex":
        def env_fn(render=render): return ModRexEnv(
            hard_reset=False, render=render, terrain_id="plane")
    elif args.env == "ant":
        def env_fn(render=render): return ModAntEnv(render)
    elif args.env == "halfcheetah":
        def env_fn(render=render): return ModHalfCheetahEnv(render)
    elif args.env == "spot":
        def env_fn(render=render): return spotGymEnv(
            hard_reset=False, render=render)

    os.makedirs(save_path, exist_ok=True)
    print("#########################################################")
    print(f"Device: {DEV}")
    print(f"Saving checkpoints in: {save_path}")
    print(f"Experiment Name: {exp_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(
        f"Using Learning Rates:: Agent: {args.lr_agent } Critic: {args.lr_critic}")
    print(f"Using env: {args.env}")
    print(f"Repeat steps: {args.repeat_steps}")
    print(f"random control: {random_control}")
    print("#########################################################")

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
    num_eps = 0
    last_ep_len = 1
    last_ep_r = 0.
    last_ep_r_survive, last_ep_r_dist, last_ep_r_speed, last_ep_r_dir, last_ep_r_angular_disp, last_ep_cost_energy = 0, 0, 0, 0, 0, 0
    for _ in pbar:
        step_idx += 1
        agent.actor_net.noise_eps = MAX_NOISE_EPSILON - \
            (MAX_NOISE_EPSILON - MIN_NOISE_EPSILON) * \
            (step_idx / STEPS_TO_REDUCE_NOISE)
        agent.actor_net.noise_eps = max(
            agent.actor_net.noise_eps, MIN_NOISE_EPSILON)
        rewards, ep_lens, steps = exp_src.step_env(
            actor_net=agent.actor_net, controls=controls, repeat_steps=repeat_steps)
        if len(rewards) > 0:
            last_ep_r, last_ep_r_comps = rewards[-1]
            last_ep_r_survive, last_ep_r_dist, last_ep_r_speed, last_ep_r_dir, last_ep_r_angular_disp, last_ep_cost_energy = last_ep_r_comps
            last_ep_len = ep_lens[-1]
            num_eps += 1

        # generate new controls for each env
        if step_idx % 100 == 0:  # new controls after every 100 steps
            if random_control:
                # thetas = np.random.rand(
                    # NUM_ENVS_PER_PROC) * math.tanh(step_idx / 1e5) * 0.5 
                # thetas = (2 * thetas) - math.tanh(step_idx / 1e5) 
                vels = np.random.rand(NUM_ENVS_PER_PROC) * \
                    math.tanh(num_eps / 1e5) * 0.5
                vels = 1 - vels
                thetas = np.zeros(NUM_ENVS_PER_PROC)
                angular_vels = np.random.rand(
                    NUM_ENVS_PER_PROC) * math.tanh(step_idx / 1e5) * 0.2 
                angular_vels = (2 * angular_vels) - math.tanh(step_idx / 1e5) * 0.2
                controls = [(v, t, w) for (v, t, w) in zip(vels, thetas, angular_vels)]
                for env, new_control in zip(exp_src.envs, controls):
                    env.update_targ(new_control)

        trajectory += steps
        if step_idx % 100 == 0:
            writer.add_scalar("epsiode_reward/r_ep", float(last_ep_r), step_idx)
            writer.add_scalar("episode_reward/r_survive", last_ep_r_survive, step_idx)
            writer.add_scalar("episode_reward/r_dist", last_ep_r_dist, step_idx)
            writer.add_scalar("episode_reward/r_speed", last_ep_r_speed, step_idx)
            writer.add_scalar("episode_reward/r_dir", last_ep_r_dir, step_idx)
            writer.add_scalar("episode_reward/r_angular_disp", last_ep_r_angular_disp, step_idx)
            writer.add_scalar("episode_reward/cost_energy", last_ep_cost_energy, step_idx)
            writer.add_scalar('control/vel', controls[0][0], step_idx,)
            writer.add_scalar('control/theta', controls[0][1], step_idx,)
            writer.add_scalar('control/angular_vel', controls[0][2], step_idx,)

        if step_idx % TEST_ITERS == 0:
            # checkpoint if best_r beaten
            test_r, test_ep_len = test_net(
                agent.actor_net, test_exp_src)
            writer.add_scalar("test/test_reward", test_r, step_idx)
            writer.add_scalar("test/test_steps", test_ep_len, step_idx)
            if best_r_mean is None or test_r >= best_r_mean:
                best_r_mean = test_r
                print(
                    f"best_r updated: {best_r_mean : 0.4f}. Saving checkpoint")
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
        old_logprob_v = calc_log_prob(
            mu_v, agent.actor_net.logstd, traj_actions_v)
        # logstd_v = 0s because current actor only returns a single action value with 0 variance, and not a probability distribution

        # normalize advantages
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / \
            torch.std(traj_adv_v)
        old_logprob_v = old_logprob_v[:-1].detach()

        # drop last entry from the trajectory, as our adv and ref value calculated without it
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
        writer.add_scalar("ppo/advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("ppo/values", traj_ref_v.mean().item(), step_idx)
        writer.add_scalar("ppo/loss_policy", sum_loss_policy /
                          count_steps, step_idx)
        writer.add_scalar("ppo/loss_value", sum_loss_value / count_steps, step_idx)
