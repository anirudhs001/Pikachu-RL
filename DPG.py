#%%
import os
from statistics import mean
import time
from tkinter import W
from tkinter.tix import MAX
from turtle import forward
import gym
import pybullet_envs
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
import sys
np.random.seed(42)
torch.manual_seed(42)

# %%
ENV_NAME = "MinitaurBulletEnv-v0"
# ENV_NAME = "MinitaurTrottingEnv-v0"
BATCH_SIZE = 128
LR_AGENT = 5e-5
LR_CRITIC = 5e-5
BUFFER_MAXLEN = 100_000
REPLAY_INITIAL = 10_000
REPLAY_START_SIZE = 1_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEV)
REPEAT_STEPS = 5
MAX_EPSILON = 1.0
MIN_EPSILON = 1e-3
GAMMA = 0.99
ENERGY_WEIGHT = 1e-3
MODEL_SAVE_PATH = "models/"
HEIGHT = 0.15
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# %%

def getAnglesFromMatrix(rot_mat):
    O_x = np.arctan2(rot_mat[3*2+1], rot_mat[3*2+2])
    O_y = np.arctan2(-rot_mat[3*2], np.sqrt(rot_mat[3*2+1]**2+rot_mat[3*2+2]**2))
    O_z = np.arctan2(rot_mat[3*1], rot_mat[0])
    return (O_x, O_y, O_z)

def pos_n_ori(env):
    '''
    env : Gym env
    '''
    # current position and orientation
    pos = env.minitaur.GetBasePosition()
    # pos is 3-tuple [x,y,z]
    rot = env.minitaur.GetBaseOrientation()
    rot_mat = pybullet.getMatrixFromQuaternion(rot)
    rot_ang = getAnglesFromMatrix(rot_mat)
    # rot_ang is 3-tuple [alpha, beta, gamma]
    return pos, rot_ang


def get_reward(control, pos1, ori1, pos2, ori2, torque, vel):
    '''
    control : desired movement of the bot
        0 : front
        1 : right
        2 : back
        3 : left
        4 : turn cw
        5 : turn ccw
    '''
    del_x = pos2[0] - pos1[0]
    del_y = pos2[1] - pos1[1]
    del_d = np.sqrt(del_x**2 + del_y**2)
    # theta = np.arctan2(del_y, del_x)
    # del_theta = theta - ori1[2]
    del_gamma = ori2[2] - ori1[2]
    # del_f = del_d * np.cos(del_theta)
    # del_l = del_d * np.sin(del_theta)
    # del_b = -del_f
    # del_r = -del_l
    energy_r = 0.
    r = 0.
    if ENERGY_WEIGHT > 0:
        for t, v in zip(torque, vel):
            energy_r += ENERGY_WEIGHT * -np.abs(np.dot(t, v))
    r += energy_r
    if control == 0:
        # r = del_f - np.abs(del_l) - np.abs(del_gamma)
        # r = del_f
        r += 5e1 * del_x - np.abs(del_y) / 10 - np.abs(del_gamma) / 10
    elif control == 1:
        # r = del_r - np.abs(del_f) - np.abs(del_gamma)
        # r = del_r
        r += 5e1 * -del_y - np.abs(del_x) / 10 - np.abs(del_gamma) / 10
    elif control == 2:
        # r = del_b - np.abs(del_l) - np.abs(del_gamma)
        # r = del_b
        r += 5e1 * -del_x - np.abs(del_y) / 10 - np.abs(del_gamma) / 10
    elif control == 3:
        # r = del_l - np.abs(del_f) - np.abs(del_gamma)
        # r = del_l
        r += 5e1 * del_y - np.abs(del_x) / 10 - np.abs(del_gamma) / 10
    elif control == 4:
        # r = -del_gamma - np.abs(del_f) - np.abs(del_l)
        r += 5e1 * -del_gamma - np.abs(del_d) / 10
    elif control == 5:
        # r = del_gamma - np.abs(del_f) - np.abs(del_l)
        r += 5e1 * del_gamma - np.abs(del_d) / 10
    return r


#### Networks

# %%
class ResLin(nn.Module):
    def __init__(self, in_n, out_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_n, out_n),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        x = x + self.net(x)
        return x


class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int, dev="cuda") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 400),
            nn.LeakyReLU(0.1),
            # ResLin(400, 400),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.1),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.1),
            nn.Linear(400, n_act),
            nn.Tanh()
        )
        self.dev = dev

    def forward(self, x):
        '''
        x : observation
        '''
        out = self.net(x)
        return out


class Critic_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''
    def __init__(self, n_obs: int, n_act: int, dev="cuda") -> None:
        super().__init__()
        self.obs_head = nn.Sequential(
            nn.Linear(n_obs, 400),
            nn.LeakyReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(400+n_act, 400),
            nn.LeakyReLU(0.1),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.1),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.1),
            # ResLin(800, 800),
            nn.Linear(400, 1),
        )
        self.dev = dev

    def forward(self, obs, act):
        obs_enc = self.obs_head(obs)
        out = torch.cat([obs_enc, act], dim=1)
        out = self.net(out)
        return out


class Agent():
    def __init__(self, n_obs, n_act, dev=DEV, rep_steps=REPEAT_STEPS, batch_size=BATCH_SIZE, epsilon= MAX_EPSILON) -> None:
        ## REPLAY BUFFER
        # 1. always keep some old experiences in replay_buf_initial to prevent catastrophic forgetting.
        self.buff_maxlen = BUFFER_MAXLEN
        self.replay_buf_initial = collections.deque(maxlen = REPLAY_INITIAL)
        self.replay_buf= collections.deque(maxlen = self.buff_maxlen - REPLAY_INITIAL)

        ## NETWORKS
        self.action_net = Agent_net(n_obs, n_act, DEV).to(DEV)
        self.action_net_targ = Agent_net(n_obs, n_act).to(DEV).requires_grad_(False)
        self.critic_net = Critic_net(n_obs, n_act).to(DEV)
        self.critic_net_targ = Critic_net(n_obs, n_act).to(DEV).requires_grad_(False)
        self.update_targ_nets(alpha = 0.)
        self.actor_optim = optim.Adam(self.action_net.parameters(), lr=LR_AGENT)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=LR_CRITIC)
        self.mse_loss = nn.MSELoss()

        ## EPISODE and STATE variables
        self.dev = DEV
        self.objective = nn.MSELoss()
        self.batch_size = batch_size
        self.rep_steps = rep_steps
        self.curr_ep_r = 0.
        self.epsilon = epsilon
        self.last_pos_n_ori = None

    def add_transn(self, control, s, s_next, r, a, is_done):
        '''Add new transn tuple in replay buffer'''
        if len(self.replay_buf_initial) < self.replay_buf_initial.maxlen:
            self.replay_buf_initial.append((control, s, s_next, r, a, is_done))
        else:
            self.replay_buf.append((control, s, s_next, r, a, is_done))


    @torch.no_grad()
    def move_k_steps(self, s, env, control, ep_len = 1, k=None):
        s_new = s
        s_new = torch.Tensor(s_new).unsqueeze(0).float().to(self.dev)
        a = self.action_net(s_new).squeeze().cpu().numpy()
        a += self.epsilon * np.random.normal(size=a.shape)
        a = np.clip(a, -1, 1)
        torque, vel = [], []
        # print(ep_len)
        # if ep_len == 1:
        self.last_pos_n_ori = pos_n_ori(env)
        if not k:
            k = self.rep_steps
        for i in range(k):
            s_new, _, is_done, _ = env.step(a)
            torque.append(env.minitaur.GetMotorTorques())
            vel.append(env.minitaur.GetMotorVelocities())

            if is_done: 
                break
        pos2, rot2 = pos_n_ori(env)
        r = get_reward(control, self.last_pos_n_ori[0], self.last_pos_n_ori[1], pos2, rot2, torque, vel)
        # r *= np.log(4 * ep_len)
        self.curr_ep_r += r
        curr_ep_r = self.curr_ep_r
        if is_done:
            s_new = env.reset()
            self.curr_ep_r = 0.
        # self.last_pos_n_ori = pos2, rot2
        return s, s_new, r, a, is_done, curr_ep_r


    def get_batch(self, batch_size=BATCH_SIZE):
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
        if len(self.replay_buf) > batch_size - batch_size//3:
            get_batch_from_buff( self.replay_buf_initial, batch_size // 3, control, s, s_next, r, a, is_done)
            get_batch_from_buff( self.replay_buf, batch_size - batch_size // 3, control, s, s_next, r, a, is_done)
        else:
            get_batch_from_buff(self.replay_buf_initial, batch_size, control, s, s_next, r, a, is_done)
            
        s = torch.FloatTensor(s).to(self.dev)
        s_next = torch.FloatTensor(s_next).to(self.dev)
        r = torch.FloatTensor(r).to(self.dev)
        a = torch.FloatTensor(a).to(self.dev)
        is_done = torch.BoolTensor(is_done).to(self.dev)
        return control, s, s_next, r, a, is_done
    

    def train_iter(self, batch_size=None, rep_steps=None):
        if len(self.replay_buf_initial) + len(self.replay_buf) < REPLAY_START_SIZE:
            return None, None

        if rep_steps is None:
            rep_steps = self.rep_steps
        if batch_size is None:
            batch_size = self.batch_size
        control, s, s_next, r, a, is_done = self.get_batch(batch_size)
        # print(s.shape, s_next.shape, r.shape, a.shape, is_done.shape)

        #CRITIC
        self.critic_optim.zero_grad()
        q_m = self.critic_net(s, a)
        a_next = self.action_net(s_next)
        _q_t = self.critic_net_targ(s_next, a_next)
        _q_t[is_done] = 0.
        q_t = r.unsqueeze(dim=-1) + ((GAMMA**rep_steps) * _q_t)  
        loss_critic = self.mse_loss(q_m, q_t.detach())
        loss_critic.backward()
        self.critic_optim.step()

        # ACTOR
        self.actor_optim.zero_grad() 
        a_m = self.action_net(s)
        loss_actor = -self.critic_net(s, a_m).mean()
        loss_actor.backward()
        self.actor_optim.step()

        return loss_actor, loss_critic
        
    def save_net(self, epoch, path="."):
        model_state = {
            "epoch": epoch,
            "action_model": self.action_net.state_dict(),
            "action_optim": self.actor_optim.state_dict(),
            "critic_model": self.critic_net.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }
        path = f"{path}/model_{epoch}.pt"
        torch.save(model_state, path)


    @torch.no_grad()
    def update_targ_nets(self, alpha):
        def update_targ_net(net, targ_net):
            net_dict = net.state_dict()
            targ_net_dict = targ_net.state_dict()
            for k, v in net_dict.items():
                targ_net_dict[k] = targ_net_dict[k] * alpha + v * (1-alpha)
            targ_net.load_state_dict(targ_net_dict)

        # update_targ_net(self.action_net, self.action_net_targ)
        update_targ_net(self.critic_net, self.critic_net_targ)
        

    def load_net(self, fname):
        model_state = torch.load(fname, map_location=self.dev)
        self.net.load_state_dict(model_state['model'])
        self.optimizer.load_state_dict(model_state['optim'])
        return model_state["epoch"]



## Training Loop

# %%

if __name__ == "__main__":
    writer = SummaryWriter(comment=f"{ENV_NAME}_ddpg")
    env = gym.make(ENV_NAME)
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
    )
    s = env.reset()
    pbar = tqdm()
    i = 0
    ep_len = 0
    last_ep_r = 0.
    last_ep_len = 0
    best_r = None
    last_10_r = collections.deque(maxlen=10)
    sum_r = 0.

    control = None
    random_control = True
    if len(sys.argv) > 1:
        print(sys.argv)
        control = int(sys.argv[1])
        random_control = False
        print(f"\n##################################\nCONTROL : {control :1d} random_control : {random_control} \n##################################")

    while(True):
        i += 1
        ep_len += 1
        if random_control and ep_len == 1:
            control = random.choice([0,2,4,5])
        # gather exp
        s, s_next, r, a, is_done, ep_r = agent.move_k_steps(s, env, control, ep_len, k=REPEAT_STEPS)
        agent.add_transn(control, s, s_next, r, a, is_done)
        s = s_next
        if is_done:
            last_ep_r = ep_r
            last_ep_len = ep_len
            ep_len = 0
            if len(last_10_r) == last_10_r.maxlen:
                sum_r -= last_10_r.popleft() 
            sum_r += last_ep_r
            last_10_r.append(last_ep_r)
            # print("average r", sum_r / len(last_10_r), "best_r", best_r)
            if best_r is None or (sum_r / len(last_10_r)) > best_r:
                best_r = sum_r / len(last_10_r) 
                if random_control:
                    agent.save_net("all")
                else:
                    agent.save_net(control)
        # train
        loss_agent, loss_critic = agent.train_iter(batch_size=BATCH_SIZE, rep_steps=REPEAT_STEPS)
        agent.update_targ_nets(alpha=1-1e-3)
        agent.epsilon = MAX_EPSILON - (MAX_EPSILON - MIN_EPSILON) * (i / 10_000)
        agent.epsilon = max(agent.epsilon, MIN_EPSILON)
        
        if i % 20 == 0:
            pbar.update(20)
            if loss_agent is not None:
                pbar.set_description(f"last ep length : {last_ep_len}, last episode reward : {last_ep_r:.4f}, loss agent = {loss_agent.item():.4f}, loss critic = {loss_critic.item():.4f}")
                writer.add_scalar("episode length", last_ep_len, i)
                writer.add_scalar("episode reward", last_ep_r, i)
                writer.add_scalar("loss agent",loss_agent.item(), i)
                writer.add_scalar("loss critic", loss_critic.item(), i)
                writer.add_scalar("epsilon", agent.epsilon, i)

# %%
