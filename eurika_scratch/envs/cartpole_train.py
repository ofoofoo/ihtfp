from typing import Callable
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib import rc
import random
from torch import nn
from gym.envs.registration import registry, register
from tqdm.notebook import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class CartpoleDynamics:
    def __init__(self,
                timestep=0.02,
                m_p=0.5,
                m_c=0.5,
                l=0.6,
                g=-9.81,
                u_range=15):
        """
        Initializes the Cartpole Dynamics model with given parameters.

        Parameters:
        - timestep (float): The time step for the simulation.
        - m_p (float): Mass of the pole.
        - m_c (float): Mass of the cart.
        - l (float): Length of the pole.
        - g (float): Acceleration due to gravity. Negative values indicate direction.
        - u_range (float): Range of the control input.
        """

        self.m_p  = m_p
        self.m_c  = m_c
        self.l    = l
        self.g    = -g
        self.dt   = timestep

        self.u_range = u_range

        self.u_lb = torch.tensor([-1]).float()
        self.u_ub = torch.tensor([1]).float()
        self.q_shape = 4
        self.u_shape = 1
        self.reward_fn = self.compute_reward

    def _qdotdot(self, q, u):
        """
        Calculates the acceleration of both cart and pole as a function of the current state and control input.

        Parameters:
        - q (torch.Tensor): The current state of the system, [x, theta, xdot, thetadot].
        - u (torch.Tensor): The current control input.

        Returns:
        - torch.Tensor: Accelerations [x_dotdot, theta_dotdot] of the cart and the pole.
        """
        x, theta, xdot, thetadot = q.T

        if len(u.shape) == 2:
            u = torch.flatten(u)

        x_dotdot = (
            u + self.m_p * torch.sin(theta) * (
                self.l * torch.pow(thetadot,2) + self.g * torch.cos(theta)
            )
        ) / (self.m_c + self.m_p * torch.sin(theta)**2)

        theta_dotdot = (
            -u*torch.cos(theta) -
            self.m_p * self.l * torch.pow(thetadot,2) * torch.cos(theta) * torch.sin(theta) -
            (self.m_c + self.m_p) * self.g * torch.sin(theta)
        ) / (self.l * (self.m_c + self.m_p * torch.sin(theta)**2))

        return torch.stack((x_dotdot, theta_dotdot), dim=-1)

    def _euler_int(self, q, qdotdot):
        """
        Performs Euler integration to calculate the new state given the current state and accelerations.

        Parameters:
        - q (torch.Tensor): The current state of the system, [x, theta, xdot, thetadot].
        - qdotdot (torch.Tensor): The accelerations [x_dotdot, theta_dotdot] of the cart and the pole.

        Returns:
        - torch.Tensor: The new state of the system after a single time step.
        """

        qdot_new = q[...,2:] + qdotdot * self.dt
        q_new = q[...,:2] + self.dt * qdot_new

        return torch.cat((q_new, qdot_new), dim=-1)

    def step(self, q, u):
        """
        Performs a single step of simulation given the current state and control input.

        Parameters:
        - q (torch.Tensor or np.ndarray): The current state of the system.
        - u (torch.Tensor or np.ndarray): The current control input.

        Returns:
        - torch.Tensor: The new state of the system after the step.
        """

        # Check for numpy array
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q)
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)

        scaled_u = u * float(self.u_range)

        # Check for shape issues
        if len(q.shape) == 2:
            q_dotdot = self._qdotdot(q, scaled_u)
        elif len(q.shape) == 1:
            q_dotdot = self._qdotdot(q.reshape(1,-1), scaled_u)
        else:
            raise RuntimeError('Invalid q shape')

        new_q = self._euler_int(q, q_dotdot)

        if len(q.shape) == 1:
            new_q = new_q[0]

        return new_q

    # given q [bs, q_shape] and u [bs, t, u_shape] run the trajectories
    def run_batch_of_trajectories(self, q, u):
        """
        Simulates a batch of trajectories given initial states and control inputs over time.

        Parameters:
        - q (torch.Tensor): Initial states for each trajectory in the batch.
        - u (torch.Tensor): Control inputs for each trajectory over time.

        Returns:
        - torch.Tensor: The states of the system at each time step for each trajectory.
        """
        qs = [q]

        for t in range(u.shape[1]):
            qs.append(self.step(qs[-1], u[:,t]))

        return torch.stack(qs, dim=1)

    # given q [bs, t, q_shape] and u [bs, t, u_shape] calculate the rewards
    def compute_reward(self, q, u):
        """
        Calculates the reward for given states and control inputs.

        Parameters:
        - q (torch.Tensor or np.ndarray): States of the system.
        - u (torch.Tensor or np.ndarray): Control inputs applied.

        Returns:
        - torch.Tensor: The calculated rewards for the states and inputs.
        """

        return ...

class CartpoleGym(gym.Env):
    def __init__(self, timestep_limit=200):
        """
        Initializes the Cartpole environment with a specified time step limit.

        Parameters:
        - timestep_limit (int): The maximum number of timesteps for each episode.

        Sets up the dynamics model and initializes the simulation state.
        """
        self.dynamics = CartpoleDynamics()

        self.timestep_limit = timestep_limit
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
        - np.ndarray: The initial state of the environment.
        """

        self.q_sim = np.zeros(4)
        self.timesteps = 0

        self.traj = [self.get_observation()]

        return self.traj[-1]

    def get_observation(self):
        """
        Retrieves the current state of the environment.

        Returns:
        - np.ndarray: The current state of the simulation.
        """

        return self.q_sim

    def step(self, action):
        """
        Executes one time step within the environment using the given action.

        Parameters:
        - action (np.ndarray): The action to apply for this timestep.

        Returns:
        - Tuple[np.ndarray, float, bool, dict]: A tuple containing the new state, the reward received,
        a boolean indicating whether the episode is done, and an info dictionary.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]

        new_q = self.dynamics.step(
            self.q_sim, action
        )

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        reward = self.dynamics.reward(
            new_q, action
        ).numpy()

        self.q_sim = new_q.numpy()
        done = self.is_done()

        self.timesteps += 1

        self.traj.append(self.q_sim)

        return self.q_sim, reward, done, {}

    def is_done(self):
        """
        Checks if the episode has finished based on the timestep limit.

        Returns:
        - bool: True if the episode is finished, False otherwise.
        """
        # Kill trial when too much time has passed
        if self.timesteps >= self.timestep_limit:
            return True

        return False

    def plot_func(self, to_plot, i=None):
        """
        Plots the current state of the cartpole system for visualization.

        Parameters:
        - to_plot (matplotlib.axes.Axes or dict): Axes for plotting or a dictionary of plot elements to update.
        - i (int, optional): The index of the current state in the trajectory to plot.
        """
        def _square(center_x, center_y, shape, angle):
            trans_points = np.array([
                [shape[0], shape[1]],
                [-shape[0], shape[1]],
                [-shape[0], -shape[1]],
                [shape[0], -shape[1]],
                [shape[0], shape[1]]
            ]) @ np.array([
                [np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)]
            ]) + np.array([center_x, center_y])

            return trans_points[:, 0], trans_points[:, 1]

        if isinstance(to_plot, Axes):
            imgs = dict(
                cart=to_plot.plot([], [], c="k")[0],
                pole=to_plot.plot([], [], c="k", linewidth=5)[0],
                center=to_plot.plot([], [], marker="o", c="k",
                                        markersize=10)[0]
            )

            x_width = max(1,max(np.abs(t[0]) for t in self.traj) * 1.3)

            # centerline
            to_plot.plot(np.linspace(-x_width, x_width, num=50), np.zeros(50),
                        c="k", linestyle="dashed")

            # set axis
            to_plot.set_xlim([-x_width, x_width])
            to_plot.set_ylim([-self.dynamics.l*1.2, self.dynamics.l*1.2])

            return imgs

        curr_x = self.traj[i]

        cart_size = (0.15, 0.1)

        cart_x, cart_y = _square(curr_x[0], 0.,
                                cart_size, 0.)

        pole_x = np.array([curr_x[0], curr_x[0] + self.dynamics.l
                        * np.cos(curr_x[1]-np.pi/2)])
        pole_y = np.array([0., self.dynamics.l
                        * np.sin(curr_x[1]-np.pi/2)])

        to_plot["cart"].set_data(cart_x, cart_y)
        to_plot["pole"].set_data(pole_x, pole_y)
        to_plot["center"].set_data(self.traj[i][0], 0.)

    def render(self, mode="human"):
        """
        Renders the current state of the environment using a matplotlib animation.

        This function creates a matplotlib figure and uses the plot_func method to update the figure with the current
        state of the cartpole system at each timestep. The animation is created with the FuncAnimation class and is
        configured to play at a specified frame rate.

        Parameters:
        - mode (str): The mode for rendering. Currently, only "human" mode is supported, which displays the animation
        on screen.

        Returns:
        - matplotlib.animation.FuncAnimation: The animation object that can be displayed in a Jupyter notebook or
        saved to file.
        """
        self.anim_fig = plt.figure()

        self.axis = self.anim_fig.add_subplot(111)
        self.axis.set_aspect('equal', adjustable='box')

        imgs = self.plot_func(self.axis)
        _update_img = lambda i: self.plot_func(imgs, i)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani = FuncAnimation(
            self.anim_fig, _update_img, interval=self.dynamics.dt*1000,
            frames=len(self.traj)-1
        )

        plt.close()

        return ani

    @property
    def action_space(self):
        """
        Defines the action space of the environment using a Box space from OpenAI Gym.

        The action space is defined based on the lower and upper bounds for the control input specified in the
        dynamics model. This allows for a continuous range of actions that can be applied to the cartpole system.

        Returns:
        - gym.spaces.Box: The action space as a Box object, with low and high bounds derived from the dynamics model's
        control input bounds.
        """
        return gym.spaces.Box(low=self.dynamics.u_lb.numpy(), high=self.dynamics.u_ub.numpy())

    @property
    def observation_space(self):
        """
        Defines the observation space of the environment using a Box space from OpenAI Gym.

        The observation space is defined with no bounds on the values, representing the position and velocity of the
        cart and the angle and angular velocity of the pole. This space allows for any real-valued vector of
        positions and velocities to be a valid observation in the environment.

        Returns:
        - gym.spaces.Box: The observation space as a Box object, with low and high bounds set to negative and
        positive infinity, respectively, for each dimension of the state vector.
        """
        return gym.spaces.Box(
            low= np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf,   np.inf,  np.inf,  np.inf])
        )

env_name = 'CartpoleSwingUp-v0'
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f'{__name__}:CartpoleGym',
)


def train_cartpole(reward_fn: Callable, pretrained_policy_dict = None):
    env = gym.make(env_name)
    env.dynamics.reward_fn = reward_fn
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100000)
    model.save("ppo_cartpole")
    return model
