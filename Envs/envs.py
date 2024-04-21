import math
import numpy as np
import collections
# import mujoco
import mujoco
from collections import OrderedDict
import time

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

np.set_printoptions(linewidth=150, precision=3, suppress=True)
np.random.seed(42)


class ModAntEnv:
    """
    The bot is about 300cm long = 0.3 in sim's dimensions
    """

    def __init__(self, render=False) -> None:
        if render:
            render_mode = "human"
        else:
            render_mode = None
        self.env = gym.make("Ant-v4", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.metadata = self.env.metadata
        self.targ_dist = 0.1
        self.targ = 0.1, 0
        self.speed_targ, self.theta_targ, self.ang_speed = 1, 0, 0

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

    def get_curr_state(self):
        x, y, z = self.env.get_body_com("torso")[:3].copy()
        ori_mat = self.env.data.body("torso").xmat.copy()
        angles_curr = self.get_angles_from_matrix(ori_mat)
        _, _, gamma = angles_curr
        return x, y, gamma

    def update_targ(self, control):
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

    def custom_reward(self, state_pre, state_post, actions, weights=[0, 1, 0, 0, 1, 0]):
        """
        control : (vel, theta) tuple
            vel in [0,1]
            theta in [-1, 1]
        """
        x_pre, y_pre, gamma_pre = state_pre
        x_post, y_post, gamma_post = state_post
        x_targ, y_targ = self.targ

        # Rewards
        w_survive, w_dist, w_speed, w_dir, w_angular_speed, w_cost_energy = weights
        # reward for survival
        r_survive = 1
        r_survive *= w_survive

        # distance reward
        dist_from_targ = np.linalg.norm((x_post - x_targ, y_post - y_targ))
        reached_targ = True if dist_from_targ <= 1e-3 else False
        r_dist = 1 - (dist_from_targ / max(1e-2, self.speed_targ * self.targ_dist))
        r_dist *= w_dist

        # speed reward
        curr_speed = np.linalg.norm((x_post - x_pre, y_post - y_pre))
        r_speed = curr_speed * self.speed_targ
        r_speed *= w_speed

        # direction reward
        motion_angle = np.arctan2((y_post - y_pre), (x_post - x_pre))
        theta_actual = gamma_post - motion_angle
        r_dir = math.cos(theta_actual - self.theta_targ)
        r_dir *= w_dir

        # angular speed reward
        angular_disp = gamma_post - gamma_pre
        angular_disp = (
            angular_disp
            + ((2 * np.pi) if angular_disp < -np.pi else 0)
            + ((-2 * np.pi) if angular_disp > np.pi else 0)
        )
        r_angular_disp = self.ang_speed * angular_disp
        r_angular_disp *= w_angular_speed

        # energy cost
        cost_energy = 0.5 * np.sum(actions ** 2)
        cost_energy *= w_cost_energy

        r = r_survive + r_dist + r_speed + r_dir + r_angular_disp - cost_energy
        r_comps = np.array(
            [r_survive, r_dist, r_speed, r_dir, r_angular_disp, cost_energy]
        )
        return r, r_comps, reached_targ

    def step(self, action, repeat_steps=1):
        state_pre = self.get_curr_state()
        for _ in range(repeat_steps):
            obs, _, is_done, _, _ = self.env.step(action)
            if is_done:
                break
        state_post = self.get_curr_state()

        r, r_comps, reached_targ = self.custom_reward(state_pre, state_post, action)
        return obs, r, r_comps, reached_targ, is_done
        # return obs, r, is_done, None

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()[0]


class ModAntEnv_V2(AntEnv):
    """
    The bot is about 300cm long = 0.3 in sim's dimensions
    """

    def __init__(self, render_mode=None, max_ep_length: int = None):
        super().__init__(render_mode=render_mode)

        # creating model and data for exposing site_xvel and other functions. these are different from self.model and self.data
        # self._model = mujoco.load_model_from_path(self.fullpath)
        
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        # self.sim = mujoco.MjSim(self._model)
        self.targ_dist = 0.1
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
        self.foot_names = [
            "right_front_foot_tip",
            "left_front_foot_tip",
            "right_back_foot_tip",
            "left_back_foot_tip",
        ]
        # self.foot_ids_map = {
        #     foot_name: self.sim.model.geom_name2id(foot_name)
        #     for foot_name in self.foot_names
        # }
        self.foot_ids_map = {
            foot_name : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.foot_names
        }
        # self.floor_body_idx = self.sim.model.geom_name2id("floor")  # floor body index
        self.floor_body_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.control = (1, 0)
        self.foot_height_target = 0.1
        # observation space with 8 joint angles with values in [-pi, pi]
        # 2 control values + 3 * size 8 joint angles + 3 * size 6 values from gyro and accelerometer + 2 * size 8 actions
        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(len(self.control) + 24 + 18 + 16,), dtype=np.float32
        )
        self.seed = lambda x: 0
        self.max_ep_length = max_ep_length
        self.ep_length = 0

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
        return x, y, alpha, beta, gamma

    def get_foot_velocity(self, foot_name):
        # Get the body index of the foot
        foot_idx = self.foot_ids_map[foot_name]
        # Get the linear velocity of the foot
        linear_velocity = self.data.geom_xpos[foot_idx]
        # this should be right. not adding angular velocity because this directly gives the velocity of the tip
        velocity = np.linalg.norm(linear_velocity)

        return velocity

    def forward_reward(self, state_pre, state_post, action):
        pass

    def custom_reward(self, state_pre, state_post, actions):
        """
        vel_control : { -1, 0, 1 }
            -1 : backward
            0 : no movement
            1 : forward
        rot_control : { -1, 0, 1 }
            -1 : left
            0 : no rotation
            1 : right
        """
        ## some reward components adapted from the solo12 paper
        # [Controlling the Solo12 Quadruped Robot with Deep Reinforcement Learning][https://hal.laas.fr/hal-03761331]
        vel_control, rot_control = (
            self.control[0],
            self.control[1],
        )
        x_pre, y_pre, alpha_pre, beta_pre, gamma_pre = state_pre
        x_post, y_post, alpha_post, beta_post, gamma_post = state_post

        # Rewards
        w_survive = 1
        w_rot = 1e2
        w_vel = 3e3
        w_cost_energy = 0
        w_jitter1 = 0
        w_jitter2 = 0
        w_torso_stability = 0
        w_slip = 0
        w_clearance = 0

        r_tot = 0
        # reward for survival
        r_survive = 1 * w_survive

        theta_rot = gamma_post - gamma_pre
        # print(gamma_post, gamma_pre, theta_rot, rot_control)
        theta_rot = (
            theta_rot + 2 * np.pi
            if theta_rot < -np.pi
            else theta_rot - 2 * np.pi
            if theta_rot > np.pi
            else theta_rot
        )

        dist = np.sqrt((x_post - x_pre) ** 2 + (y_post - y_pre) ** 2)
        # motion_dir = np.arctan2(y_post - y_pre, x_post - x_pre)
        # angle_diff = motion_dir - gamma_pre
        # dist_comp = dist * (math.e**np.cos(angle_diff))
        # dist_comp = dist * np.cos(angle_diff)

        # rotate left/right
        if rot_control in [1, -1] and vel_control == 0:
            r_rot = theta_rot * rot_control * w_rot
            # cost_dist = dist * w_vel * 1e-2
            cost_dist = 0
            r_tot += r_rot - cost_dist

        # walk forward/backward
        elif vel_control in [1, -1] and rot_control == 0:
            # cost_rot = np.abs(theta_rot) * w_rot * 1e-2
            cost_rot = 0
            r_dist = (x_post - x_pre) * vel_control * w_vel
            r_tot = r_dist - cost_rot

        # stop
        elif vel_control == 0 and rot_control == 0:
            cost_rot = np.abs(theta_rot) * w_rot * 1e-1
            cost_dist = dist * w_vel * 1e-1
            r_tot = -cost_rot - cost_dist

        # energy cost
        cost_energy = 0.5 * np.sum(actions ** 2) * w_cost_energy

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

        # torso stability
        cost_torso_stability = (
            np.abs(alpha_post - alpha_pre) ** 2 + np.abs(beta_post - beta_pre) ** 2
        ) * w_torso_stability

        # floor slip penalty
        # Get the index of the floor body
        # List of bodies to check for contact with the floor
        contacts = [
            (c.geom1, c.geom2)
            for c in self.data.contact
            if c.geom1 == self.floor_body_idx or c.geom2 == self.floor_body_idx
        ]
        # remove duplicate contacts. keep order
        contacts = list(OrderedDict.fromkeys(contacts))
        contact_mask = []
        for foot_name, foot_id in self.foot_ids_map.items():
            for c in contacts:
                if foot_id in c:
                    contacts[contacts.index(c)] = (c[1], c[0])
                    contact_mask.append(1)
                    break
            contact_mask.append(0)
        # foot end velocities
        # foot_vels = [self.get_foot_velocity(
        #     foot_name) for foot_name in self.foot_names]
        foot_vel_map = {
            foot_name: self.get_foot_velocity(foot_name)
            for foot_name in self.foot_names
        }
        # foot slip penalty
        cost_slip = (
            np.sum(
                [
                    foot_vel_map[foot_name] * contact_mask[i]
                    for i, foot_name in enumerate(self.foot_names)
                ]
            )
            * w_slip
        )
        # foot clearance penalty
        cost_clearance = (
            np.sum(
                [
                    (
                        (
                            self.data.geom_xpos[self.foot_ids_map[foot_name]][2]
                            - self.foot_height_target
                        )
                        ** 2
                    )
                    * np.sqrt(foot_vel_map[foot_name])
                    for foot_name in self.foot_names
                ]
            )
            * w_clearance
        )

        # total reward
        r_tot = (
            r_tot
            + r_survive
            - cost_energy
            - cost_jitter
            - cost_torso_stability
            - cost_slip
            - cost_clearance
        )
        r_comps = [
            r_tot,
            r_survive,
            cost_energy,
            cost_jitter,
            cost_torso_stability,
            cost_slip,
            cost_clearance,
        ]
        # print('r_comps: ', r_comps)
        return r_tot, r_comps

    def step(self, action, repeat_steps=2, control=None):
        self.ep_length += repeat_steps
        if control:
            self.control = control
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

        # r = r - 50 if _is_done else r
        is_done = is_done or (self.max_ep_length is not None and self.ep_length >= self.max_ep_length) 
        is_truncated = self.ep_length >= self.max_ep_length
        # return obs, (r, r_comps), is_done, {'episode':None}
        return obs, r, is_done, is_truncated, {"episode": None}

    def set_control(self, control):
        self.control = control
        return self.control

    def reset(self, seed=None, options=None):
        self.ep_length = 0
        # clear both repositories
        self.jnt_angles_repository.clear()
        self.actions_repository.clear()

        # update control
        control = np.random.randint(low=0, high=4)
        self.control = (
            [1, 0]
            if control == 0
            else [-1, 0]
            if control == 1
            else [0, 1]
            if control == 2
            else [0, -1]
        )

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
        # Specify the desired joint angles
        desired_joint_angles = np.random.uniform(size=8, low=-0.4, high=0.4)
        self.init_qpos[-8:] = desired_joint_angles
        super().reset()  # reset will set these angles

        obs = np.concatenate((control, jnt_angles_flat, sensor_data_flat, actions_flat))
        return obs, True

    # def set_joint_poses(leg_name, joint_angle):


if __name__ == "__main__":
    env = ModAntEnv_V2(render_mode="human")
    # get initial observation and size
    obs = env.reset()
    print("obs", obs, obs.shape)
    step = 0
    idx = 0
    forward_actions = [
        np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])]
    rotate_actions = [
        np.array([1, 1, -1, -1, -1, -1, 1, 1]), np.array([-1, -1, 1, 1, 1, 1, -1, -1])]
    actions_rep = [forward_actions, rotate_actions]
    env.control = (1,0)
    while True:
        step += 1
        if step >= len(actions_rep[idx]) * 30:
            step = 0
        # action = np.zeros(8)
        action = actions_rep[idx][step // 30]
        obs, r, is_done, _ = env.step(action, repeat_steps=2)
        print("action: ", action, action.shape)
        print("obs: ", obs, obs.shape)
        print("r: ", r)
        env.render()
        time.sleep(0.1)

    # action_seq = [
    #     [0.5, 0.666, 0.5, 0.666, 0.333, 0.5, 0.333, 0.5],
    # ]

    # while True:
    #     step += 1
    #     action = action_seq[(step // 30) % len(action_seq)]
    #     action = np.array(action)
    #     obs, r, is_done, _ = env.step(action, repeat_steps=2)
    #     # print("action: ", action, action.shape)
    #     # print("obs: ", obs, obs.shape)
    #     # print("r: ", r)
    #     env.render()
    #     time.sleep(0.1)
