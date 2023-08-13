# %%
import random
# from statistics import mean
# import time
# import gym
# import pybullet_envs
# import pybullet
import argparse
import collections
from tensorboardX import SummaryWriter
import numpy as np
# import random
from tqdm import tqdm
# import collections
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from rex_gym.envs import rex_gym_env
# import sys
# sys.path.append("..")
# from spotmicro.spotmicro.spot_gym_env import spotGymEnv
torch.manual_seed(43)
np.random.seed(43)


BATCH_SIZE = 100
LR = 1e-4
BUF_SIZE = 1_000_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10_000
TRAIN_START_STEP = 1_000
# print(DEV)
REPEAT_STEPS = 1
MAX_EP_LEN = 2_000
GAMMA = 0.99
SOFT_UPDATE_ALPHA = 0.995
ALPHA = 0.2
N = 10
G = 20
M = 2

ENERGY_WEIGHT = 1e-5
HEIGHT = 1.2


class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given
    state
    '''

    def __init__(self, n_obs: int, n_act: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh()
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mu = nn.Sequential(
            nn.Linear(64, n_act),
        )
        self.log_std = nn.Sequential(
            nn.Linear(64, n_act),
        )
        self.dev = DEV

    def forward(self, x):
        '''
        x : observation
        '''
        # x = x.to(self.dev)
        x = self.l1(x)
        x = x + self.l2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_pi_a = (dist.log_prob(u) -
                    (2 * (np.log(2) - u - F.softplus(-2 * u)))).sum()
        # source : https://github.com/XinJingHao/SAC-Continuous-Pytorch/blob/07b6a5d0cb70f11db793ae39b1024a17379b5534/SAC.py#L51
        return a, log_pi_a


class Q_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''

    def __init__(self, n_obs: int, n_act: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_obs+n_act, 64),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.dev = DEV

    def forward(self, obs, act):
        # obs = obs.to(self.dev)
        # act = act.to(self.dev)
        x = torch.cat([obs, act], dim=1)
        q = self.l1(x)
        q = q + self.l2(q)
        q = self.l3(q)
        return q


class REDQ_Agent(nn.Module):
    def __init__(self, obs_space, act_space, lr):
        super().__init__()
        n_obs = obs_space.shape[0]
        n_act = act_space.shape[0]
        self.policy_net = Agent_net(n_obs, n_act).to(DEV)
        self.q_nets = [Q_net(n_obs, n_act).to(DEV) for _ in range(N)]
        self.targ_q_nets = [Q_net(n_obs, n_act).to(DEV) for _ in range(N)]
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_net_optims = [optim.Adadelta(
            q_net.parameters(), lr=lr) for q_net in self.q_nets]
        for q_net, targ_q_net in zip(self.q_nets, self.targ_q_nets):
            self.update_targ_net(q_net, targ_q_net, 0)

        self.buff_maxlen = BUF_SIZE
        self.replay_buf_initial = collections.deque(maxlen=BUF_SIZE // 10)
        self.replay_buf = collections.deque(maxlen=BUF_SIZE - BUF_SIZE // 10)

        self.curr_ep_r = 0.
        self.curr_ep_len = 0

    @torch.no_grad()
    def update_targ_net(self, net, targ_net, alpha=SOFT_UPDATE_ALPHA):
        net_dict = net.state_dict()
        targ_net_dict = targ_net.state_dict()
        for k, v in net_dict.items():
            targ_net_dict[k] = targ_net_dict[k] * \
                alpha + v * (1-alpha)
        targ_net.load_state_dict(targ_net_dict)

    def add_transn(self, control, s, s_next, r, a, is_done):
        '''Add new transn tuple in replay buffer'''
        if len(self.replay_buf_initial) < self.replay_buf_initial.maxlen:
            self.replay_buf_initial.append((control, s, s_next, r, a, is_done))
        else:
            self.replay_buf.append((control, s, s_next, r, a, is_done))

    @torch.no_grad()
    def move_k_steps(self, s, env, control=0):
        s_new = s
        s_new = torch.Tensor(s_new).unsqueeze(0).float().to(DEV)
        a = self.policy_net(s_new)[0].squeeze().cpu().numpy()
        r = 0.
        for i in range(REPEAT_STEPS):
            s_new, _r, is_done, _ = env.step(a)
            r += _r
            if is_done or self.curr_ep_len >= MAX_EP_LEN:
                break
        # r = get_reward(control, self.last_pos_n_ori[0], self.last_pos_n_ori[1], pos2, rot2, torque, vel)
        self.curr_ep_r += r
        curr_ep_r = self.curr_ep_r
        if is_done:
            s_new = env.reset()
            self.curr_ep_r = 0.
        # self.last_pos_n_ori = pos2, rot2
        return s, s_new, r, a, is_done, curr_ep_r

    def get_batch(self):
        def get_batch_from_buff(buff, batch_size, control, s, s_next, r, a, is_done):
            minibatch = random.sample(buff, batch_size)
            for sample in minibatch:
                control.append(sample[0])
                s.append(sample[1])
                s_next.append(sample[2])
                r.append(sample[3])
                a.append(sample[4])
                is_done.append(sample[5])

        control, s, s_next, r, a, is_done = [], [], [], [], [], []
        if len(self.replay_buf) > BATCH_SIZE - BATCH_SIZE//3:
            get_batch_from_buff(
                self.replay_buf_initial, BATCH_SIZE // 3, control, s, s_next, r, a, is_done)
            get_batch_from_buff(self.replay_buf, BATCH_SIZE -
                                BATCH_SIZE // 3, control, s, s_next, r, a, is_done)
        else:
            get_batch_from_buff(self.replay_buf_initial,
                                BATCH_SIZE, control, s, s_next, r, a, is_done)

        s = torch.FloatTensor(s).to(DEV)
        s_next = torch.FloatTensor(s_next).to(DEV)
        r = torch.FloatTensor(r).to(DEV)
        a = torch.FloatTensor(a).to(DEV)
        is_done = torch.BoolTensor(is_done).to(DEV)
        return control, s, s_next, r, a, is_done

    def train_iter(self):
        loss_q, loss_p = 0., 0.
        # train q_nets
        for _ in range(G):
            _, s, s_next, r, a, is_done = self.get_batch()
            # compute Q target
            with torch.no_grad():
                a_next, log_pi_a_next = self.policy_net(s_next)
                y = 1e8
                # select M q_nets randomly from N
                indx_of_nets_to_use = random.sample(range(0, N), k=M)
                q_preds = []
                for i in indx_of_nets_to_use:
                    q_preds.append(self.targ_q_nets[i](s_next, a_next))
                q_preds = torch.cat(q_preds, 1)
                min_q, _ = torch.min(q_preds, dim=1, keepdim=True)
                y = r + GAMMA * (min_q - ALPHA * log_pi_a_next)

            for q_net, q_net_optim, targ_q_net in zip(self.q_nets, self.q_net_optims, self.targ_q_nets):
                q_net_optim.zero_grad()
                q = q_net(s, a)
                loss = F.mse_loss(y, q)
                loss.backward()
                loss_q += loss.item()
                q_net_optim.step()
                # update targ nets
                self.update_targ_net(q_net, targ_q_net)
        loss_q /= G*N

        # train policy
        a_pred, log_pi_a_pred = self.policy_net(s)
        self.policy_optim.zero_grad()
        y_av = 0.
        for q_net in self.q_nets:
            y_av += q_net(s, a_pred)
        y_av /= N
        y_av = r + GAMMA * (y_av - ALPHA * log_pi_a_pred)
        loss = -y_av.mean()  # need to do gradient ascent on y_av
        loss.backward()
        loss_p = loss.item()
        self.policy_optim.step()
        return loss_p, loss_q

    def save_net(self, comment="", path="."):
        model_state = {
            "comment": comment,
            "policy_net": self.policy_net.state_dict(),
        }
        path = f"{path}/model_{comment}.pt"
        torch.save(model_state, path)

    def load_net(self, comment, path="."):
        path = f"{path}/model_{comment}.pt"
        state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict['policy_net'])


def generator():
    while True:
        yield


if __name__ == "__main__":
    print("TRAINING")
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--save_path",
                        help="Path to save checkpoints", default=".")
    parser.add_argument(
        "-c",
        "--comment",
        help="Optional comment to differentiate experiments. This will be added to both save_path and run name in tensorboard",
        default=""
    )
    parser.add_argument("-l", "--lr", help="Learning rate", default=LR)

    # Read arguments from command line
    args = parser.parse_args()

    save_path = "./"
    exp_name = "spot_REDQ"
    lr = LR
    if args.save_path:
        save_path = args.save_path
    if args.comment:
        save_path += f"_{args.comment}"
        exp_name += f"_{args.comment}"
    if args.lr:
        lr = float(args.lr)

    print(f"Saving checkpoints in: {save_path}")
    print(f"Experiment Name: {exp_name}")
    print(f"Using Learning Rate: {lr}")

    writer = SummaryWriter(comment=exp_name)
    # env = spotGymEnv(render=False, hard_reset=False)
    env = rex_gym_env.RexGymEnv(render=False, hard_reset=False, terrain_id="plane")
    redq_agent = REDQ_Agent(env.observation_space, env.action_space, lr=lr)
    s = env.reset()

    pbar = tqdm(generator())
    i = 0
    last_ep_r = 0.
    best_r = None
    best_r_i = 0
    loss_p, loss_q = None, None
    for _ in pbar:
        # gather exp
        s, s_new, r, a, is_done, curr_ep_r = redq_agent.move_k_steps(s, env)
        redq_agent.add_transn(0, s, s_new, r, a, is_done)
        if is_done:
            last_ep_r = curr_ep_r
            if best_r is None or best_r < last_ep_r:
                best_r = last_ep_r
                best_r_i = i
                redq_agent.save_net()
            pbar.set_description( f"last ep_r = {last_ep_r : .4f} best ep_r = {best_r:.4f} at {best_r_i}")

        if i >= TRAIN_START_STEP:
            # if i % 100 == 0:
            loss_p, loss_q = redq_agent.train_iter()

        if i % 100:
            writer.add_scalar("episode_reward", last_ep_r, i)
            if loss_p is not None:
                writer.add_scalar("policy_loss", loss_p, i)
                writer.add_scalar("average_value_loss", loss_q, i)
        i += 1

# %%
