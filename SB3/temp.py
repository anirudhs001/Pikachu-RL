# The environment :
import torch
import math
import numpy as np
import collections
import time
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import SubprocVecEnv
import sys

# use gymnasium for mac, gym for linux/windows
if sys.platform == "darwin":
    import gymnasium as gym
    from gymnasium.envs.mujoco.ant_v4 import AntEnv
    from gymnasium import spaces
else:
    import gym
    from gym.envs.mujoco.ant_v4 import AntEnv
    from gym import spaces

np.random.seed(43)


def vec_norm(vec):
    return sum([v ** 2 for v in vec]) ** 0.5


class ModAntEnv_V2(AntEnv):
    def __init__(self, render_mode=None) -> None:
        super().__init__(render_mode=render_mode)

        self.targ_dist = 0.1
        self.targ = 0.1, 0
        self.speed_targ, self.theta_targ, self.ang_speed = 1, 0, 0
        self.jnt_angles_repository = collections.deque(maxlen=3)
        self.actions_repository = collections.deque(maxlen=2)
        self.sensor_data_repository = collections.deque(maxlen=3)
        self.jnt_names = [
            "ankle_1",
            "ankle_2",
            "ankle_3",
            "ankle_4",
            "hip_1",
            "hip_2",
            "hip_3",
            "hip_4",
        ]
        self.jnt_adrs = [
            self.model.jnt(jnt_name).qposadr[0] for jnt_name in self.jnt_names
        ]

        self.control = (1, 0)
        self.step_idx = 0

        # observation space with 8 joint angles with values in [-pi, pi]
        # 2 control values + 3 * size 8 joint angles + 3 * size 6 values from gyro and accelerometer + 2 * size 8 actions
        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(2 + 24 + 18 + 16,), dtype=np.float32
        )
        self.seed = lambda x: 0

    def get_angles_from_matrix(self, rot_mat):
        """
        returns angles in radians
        """
        O_x = np.arctan2(rot_mat[3 * 2 + 1], rot_mat[3 * 2 + 2])
        O_y = np.arctan2(
            -rot_mat[3 * 2], np.sqrt(rot_mat[3 * 2 + 1] ** 2 + rot_mat[3 * 2 + 2] ** 2)
        )
        O_z = np.arctan2(rot_mat[3 * 1], rot_mat[0])
        return (O_x, O_y, O_z)

    def get_jnt_angles(self):
        # return super()._get_obs()
        jnt_angles = []
        for adr in self.jnt_adrs:
            jnt_angles.append(self.data.qpos[adr])
        return np.array(jnt_angles)

    def get_sensor_data(self):
        # pdb.set_trace()
        gyro_data = self.data.sensor("gyro").data.copy()
        accel_data = self.data.sensor("accel").data.copy()
        return np.concatenate([gyro_data, accel_data])

    def get_curr_pos_and_angle(self):
        x, y, z = self.get_body_com("torso")[:3].copy()
        ori_mat = self.data.body("torso").xmat.copy()
        angles_curr = self.get_angles_from_matrix(ori_mat)
        alpha, beta, gamma = angles_curr
        return x, y, gamma

    def update_targ(self, control):
        """
        control : (vel, rot) tuple
            vel in [0,1]
            rot in {-1, 0, 1}
        """
        return
        self.speed_targ, self.theta_targ, self.ang_speed = control
        self.theta_targ *= np.pi  # to radians
        curr_x, curr_y, curr_gamma = self.get_curr_state()
        targ_theta = curr_gamma + self.theta_targ
        targ_theta = (
            targ_theta
            + ((2 * np.pi) if targ_theta < -np.pi else 0)
            + ((-2 * np.pi) if targ_theta > np.pi else 0)
        )
        # distance is proportional to speed
        targ_x = curr_x + self.targ_dist * self.speed_targ * math.cos(targ_theta)
        targ_y = curr_y + self.targ_dist * self.speed_targ * math.sin(targ_theta)
        self.targ = (targ_x, targ_y)

    def custom_reward(
        self, state_pre, state_post, actions, weights=[1, 1e2, 1e3, 0, 0, 0]
    ):
        vel_control, rot_control = self.control
        x_pre, y_pre, gamma_pre = state_pre
        x_post, y_post, gamma_post = state_post

        # Rewards
        # w_survive, w_dist, w_speed, w_dir, w_angular_speed, w_cost_energy = weights
        w_survive, w_rot, w_vel, w_cost_energy, w_jitter1, w_jitter2 = weights
        # reward for survival
        r_survive = 1
        r_survive *= w_survive

        # rotation reward
        theta_rot = gamma_post - gamma_pre
        theta_rot = (
            theta_rot + 2 * np.pi
            if theta_rot < -np.pi
            else theta_rot - 2 * np.pi
            if theta_rot > np.pi
            else theta_rot
        )
        r_rot = theta_rot * rot_control
        r_rot *= w_rot

        # distance reward
        dist = np.sqrt((x_post - x_pre) ** 2 + (y_post - y_pre) ** 2)
        motion_dir = np.arctan2(y_post - y_pre, x_post - x_pre)
        angle_diff = motion_dir - gamma_pre
        # dist_comp = dist * (math.e**np.cos(angle_diff))
        dist_comp = dist * np.cos(angle_diff)
        r_dist = dist_comp if rot_control == 0 else -dist * 1e-1
        r_dist *= w_vel * vel_control

        # energy cost
        cost_energy = 0.5 * np.sum(actions ** 2)
        cost_energy *= (1 - vel_control) * w_cost_energy

        # action smoothness penalty
        cost_jitter = (
            w_jitter1 * np.sum(np.abs(actions - self.actions_repository[-1])) ** 2
            + w_jitter2
            * np.sum(
                np.abs(
                    actions
                    - 2 * self.actions_repository[-1]
                    + self.actions_repository[-2]
                )
            )
            ** 2
        )
        # total reward
        # distance travelled should be penalized

        r = r_survive + r_rot + r_dist - cost_energy - cost_jitter
        r_comps = [r_survive, r_rot, r_dist, cost_energy, cost_jitter]
        return r, r_comps

    def step(self, action, repeat_steps=2):
        self.step_idx += 1
        state_pre = self.get_curr_pos_and_angle()
        for _ in range(repeat_steps):
            self.do_simulation(action, self.frame_skip)
            is_done = self.terminated
            if is_done:
                break
        # obs = self._get_obs()
        state_post = self.get_curr_pos_and_angle()
        r, r_comps = self.custom_reward(state_pre, state_post, action)
        # print('state_pre: ', state_pre, 'state_post: ', state_post)
        # print('r_comps: ', r_comps)
        jnt_angles = self.get_jnt_angles()
        sensor_data = self.get_sensor_data()
        self.jnt_angles_repository.append(jnt_angles)
        self.actions_repository.append(action)
        self.sensor_data_repository.append(sensor_data)

        # concat jnt_angles_repository and actions_repositorys, both of which are collections.deque
        jnt_angles_flat = np.array(self.jnt_angles_repository).flatten()
        sensor_data_flat = np.array(self.sensor_data_repository).flatten()
        actions_flat = np.array(self.actions_repository).flatten()
        control = np.array(self.control)
        obs = np.concatenate((control, jnt_angles_flat, sensor_data_flat, actions_flat))

        # return obs, (r, r_comps), is_done, {'episode':None}
        return obs, r, is_done, {"episode": None}

    def reset(self, control=None):
        # clear both repositories
        self.jnt_angles_repository.clear()
        self.actions_repository.clear()
        self.step_idx = 0
        if control is not None:
            self.control = control
        else:
            # randomly sample control
            rot_control = np.random.choice([-1, 0, 1])
            # rot_control = 0
            # vel_control = np.random.choice(np.linspace(0, 1, 10)) if rot_control == 0 else 0
            vel_control = 1 if rot_control == 0 else 0
            self.control = (vel_control, rot_control)

        # make observation with 8 joint angles the same way as in step function
        for _ in range(2):
            self.jnt_angles_repository.append(np.zeros(8))
            self.actions_repository.append(np.zeros(8))
            self.sensor_data_repository.append(np.zeros(6))
        self.jnt_angles_repository.append(self.get_jnt_angles())
        self.sensor_data_repository.append(self.get_sensor_data())
        jnt_angles_flat = np.array(self.jnt_angles_repository).flatten()
        sensor_data_flat = np.array(self.sensor_data_repository).flatten()
        actions_flat = np.array(self.actions_repository).flatten()
        control = np.array(self.control)
        super().reset()
        obs = np.concatenate((control, jnt_angles_flat, sensor_data_flat, actions_flat))
        return obs


if __name__ == "__main__":
    n_steps = 100_000_000
    n_envs = 1
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[32, 32]),
        squash_output=True,
        use_expln=True,
    )

    env = make_vec_env(ModAntEnv_V2, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    learner = lambda *args, **kwargs: PPO(
        *args,
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=128,
        policy_kwargs=policy_kwargs,
        **kwargs,
    )
    model = learner(verbose=1, device="cpu", use_sde=True)
    model.learn(total_timesteps=n_steps)
