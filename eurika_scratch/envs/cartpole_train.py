from typing import Callable
import gymnasium as gym
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
from gymnasium.envs.registration import registry, register
from tqdm.notebook import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Optional, Tuple, Union
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


# class CartpoleDynamics:
#     def __init__(self,
#                 timestep=0.02,
#                 m_p=0.5,
#                 m_c=0.5,
#                 l=0.6,
#                 g=-9.81,
#                 u_range=15):
#         """
#         Initializes the Cartpole Dynamics model with given parameters.

#         Parameters:
#         - timestep (float): The time step for the simulation.
#         - m_p (float): Mass of the pole.
#         - m_c (float): Mass of the cart.
#         - l (float): Length of the pole.
#         - g (float): Acceleration due to gravity. Negative values indicate direction.
#         - u_range (float): Range of the control input.
#         """

#         self.m_p  = m_p
#         self.m_c  = m_c
#         self.l    = l
#         self.g    = -g
#         self.dt   = timestep

#         self.u_range = u_range

#         self.u_lb = torch.tensor([-1]).float()
#         self.u_ub = torch.tensor([1]).float()
#         self.q_shape = 4
#         self.u_shape = 1
#         self.reward_fn = self.compute_reward

#     def _qdotdot(self, q, u):
#         """
#         Calculates the acceleration of both cart and pole as a function of the current state and control input.

#         Parameters:
#         - q (torch.Tensor): The current state of the system, [x, theta, xdot, thetadot].
#         - u (torch.Tensor): The current control input.

#         Returns:
#         - torch.Tensor: Accelerations [x_dotdot, theta_dotdot] of the cart and the pole.
#         """
#         x, theta, xdot, thetadot = q.T

#         if len(u.shape) == 2:
#             u = torch.flatten(u)

#         x_dotdot = (
#             u + self.m_p * torch.sin(theta) * (
#                 self.l * torch.pow(thetadot,2) + self.g * torch.cos(theta)
#             )
#         ) / (self.m_c + self.m_p * torch.sin(theta)**2)

#         theta_dotdot = (
#             -u*torch.cos(theta) -
#             self.m_p * self.l * torch.pow(thetadot,2) * torch.cos(theta) * torch.sin(theta) -
#             (self.m_c + self.m_p) * self.g * torch.sin(theta)
#         ) / (self.l * (self.m_c + self.m_p * torch.sin(theta)**2))

#         return torch.stack((x_dotdot, theta_dotdot), dim=-1)

#     def _euler_int(self, q, qdotdot):
#         """
#         Performs Euler integration to calculate the new state given the current state and accelerations.

#         Parameters:
#         - q (torch.Tensor): The current state of the system, [x, theta, xdot, thetadot].
#         - qdotdot (torch.Tensor): The accelerations [x_dotdot, theta_dotdot] of the cart and the pole.

#         Returns:
#         - torch.Tensor: The new state of the system after a single time step.
#         """

#         qdot_new = q[...,2:] + qdotdot * self.dt
#         q_new = q[...,:2] + self.dt * qdot_new

#         return torch.cat((q_new, qdot_new), dim=-1)

#     def step(self, q, u):
#         """
#         Performs a single step of simulation given the current state and control input.

#         Parameters:
#         - q (torch.Tensor or np.ndarray): The current state of the system.
#         - u (torch.Tensor or np.ndarray): The current control input.

#         Returns:
#         - torch.Tensor: The new state of the system after the step.
#         """

#         # Check for numpy array
#         if isinstance(q, np.ndarray):
#             q = torch.from_numpy(q)
#         if isinstance(u, np.ndarray):
#             u = torch.from_numpy(u)

#         scaled_u = u * float(self.u_range)

#         # Check for shape issues
#         if len(q.shape) == 2:
#             q_dotdot = self._qdotdot(q, scaled_u)
#         elif len(q.shape) == 1:
#             q_dotdot = self._qdotdot(q.reshape(1,-1), scaled_u)
#         else:
#             raise RuntimeError('Invalid q shape')

#         new_q = self._euler_int(q, q_dotdot)

#         if len(q.shape) == 1:
#             new_q = new_q[0]

#         return new_q

#     # given q [bs, q_shape] and u [bs, t, u_shape] run the trajectories
#     def run_batch_of_trajectories(self, q, u):
#         """
#         Simulates a batch of trajectories given initial states and control inputs over time.

#         Parameters:
#         - q (torch.Tensor): Initial states for each trajectory in the batch.
#         - u (torch.Tensor): Control inputs for each trajectory over time.

#         Returns:
#         - torch.Tensor: The states of the system at each time step for each trajectory.
#         """
#         qs = [q]

#         for t in range(u.shape[1]):
#             qs.append(self.step(qs[-1], u[:,t]))

#         return torch.stack(qs, dim=1)

#     # given q [bs, t, q_shape] and u [bs, t, u_shape] calculate the rewards
#     def compute_reward(self, q, u):
#         """
#         Calculates the reward for given states and control inputs.

#         Parameters:
#         - q (torch.Tensor or np.ndarray): States of the system.
#         - u (torch.Tensor or np.ndarray): Control inputs applied.

#         Returns:
#         - torch.Tensor: The calculated rewards for the states and inputs.
#         """

#         return ...

#     def reward()

# class CartpoleGym(gym.Env):
#     def __init__(self, timestep_limit=200):
#         """
#         Initializes the Cartpole environment with a specified time step limit.

#         Parameters:
#         - timestep_limit (int): The maximum number of timesteps for each episode.

#         Sets up the dynamics model and initializes the simulation state.
#         """
#         self.dynamics = CartpoleDynamics()

#         self.timestep_limit = timestep_limit
#         self.reset()

#     def reset(self):
#         """
#         Resets the environment to the initial state.

#         Returns:
#         - np.ndarray: The initial state of the environment.
#         """

#         self.q_sim = np.zeros(4)
#         self.timesteps = 0

#         self.traj = [self.get_observation()]

#         return self.traj[-1]

#     def get_observation(self):
#         """
#         Retrieves the current state of the environment.

#         Returns:
#         - np.ndarray: The current state of the simulation.
#         """

#         return self.q_sim

#     def step(self, action):
#         """
#         Executes one time step within the environment using the given action.

#         Parameters:
#         - action (np.ndarray): The action to apply for this timestep.

#         Returns:
#         - Tuple[np.ndarray, float, bool, dict]: A tuple containing the new state, the reward received,
#         a boolean indicating whether the episode is done, and an info dictionary.
#         """
#         action = np.clip(action, self.action_space.low, self.action_space.high)[0]

#         new_q = self.dynamics.step(
#             self.q_sim, action
#         )

#         if not isinstance(action, torch.Tensor):
#             action = torch.tensor(action)

#         reward = self.dynamics.reward(
#             new_q, action
#         ).numpy()

#         self.q_sim = new_q.numpy()
#         done = self.is_done()

#         self.timesteps += 1

#         self.traj.append(self.q_sim)

#         return self.q_sim, reward, done, {}

#     def is_done(self):
#         """
#         Checks if the episode has finished based on the timestep limit.

#         Returns:
#         - bool: True if the episode is finished, False otherwise.
#         """
#         # Kill trial when too much time has passed
#         if self.timesteps >= self.timestep_limit:
#             return True

#         return False

#     def plot_func(self, to_plot, i=None):
#         """
#         Plots the current state of the cartpole system for visualization.

#         Parameters:
#         - to_plot (matplotlib.axes.Axes or dict): Axes for plotting or a dictionary of plot elements to update.
#         - i (int, optional): The index of the current state in the trajectory to plot.
#         """
#         def _square(center_x, center_y, shape, angle):
#             trans_points = np.array([
#                 [shape[0], shape[1]],
#                 [-shape[0], shape[1]],
#                 [-shape[0], -shape[1]],
#                 [shape[0], -shape[1]],
#                 [shape[0], shape[1]]
#             ]) @ np.array([
#                 [np.cos(angle), np.sin(angle)],
#                 [-np.sin(angle), np.cos(angle)]
#             ]) + np.array([center_x, center_y])

#             return trans_points[:, 0], trans_points[:, 1]

#         if isinstance(to_plot, Axes):
#             imgs = dict(
#                 cart=to_plot.plot([], [], c="k")[0],
#                 pole=to_plot.plot([], [], c="k", linewidth=5)[0],
#                 center=to_plot.plot([], [], marker="o", c="k",
#                                         markersize=10)[0]
#             )

#             x_width = max(1,max(np.abs(t[0]) for t in self.traj) * 1.3)

#             # centerline
#             to_plot.plot(np.linspace(-x_width, x_width, num=50), np.zeros(50),
#                         c="k", linestyle="dashed")

#             # set axis
#             to_plot.set_xlim([-x_width, x_width])
#             to_plot.set_ylim([-self.dynamics.l*1.2, self.dynamics.l*1.2])

#             return imgs

#         curr_x = self.traj[i]

#         cart_size = (0.15, 0.1)

#         cart_x, cart_y = _square(curr_x[0], 0.,
#                                 cart_size, 0.)

#         pole_x = np.array([curr_x[0], curr_x[0] + self.dynamics.l
#                         * np.cos(curr_x[1]-np.pi/2)])
#         pole_y = np.array([0., self.dynamics.l
#                         * np.sin(curr_x[1]-np.pi/2)])

#         to_plot["cart"].set_data(cart_x, cart_y)
#         to_plot["pole"].set_data(pole_x, pole_y)
#         to_plot["center"].set_data(self.traj[i][0], 0.)

#     def render(self, mode="human"):
#         """
#         Renders the current state of the environment using a matplotlib animation.

#         This function creates a matplotlib figure and uses the plot_func method to update the figure with the current
#         state of the cartpole system at each timestep. The animation is created with the FuncAnimation class and is
#         configured to play at a specified frame rate.

#         Parameters:
#         - mode (str): The mode for rendering. Currently, only "human" mode is supported, which displays the animation
#         on screen.

#         Returns:
#         - matplotlib.animation.FuncAnimation: The animation object that can be displayed in a Jupyter notebook or
#         saved to file.
#         """
#         self.anim_fig = plt.figure()

#         self.axis = self.anim_fig.add_subplot(111)
#         self.axis.set_aspect('equal', adjustable='box')

#         imgs = self.plot_func(self.axis)
#         _update_img = lambda i: self.plot_func(imgs, i)

#         Writer = animation.writers['ffmpeg']
#         writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#         ani = FuncAnimation(
#             self.anim_fig, _update_img, interval=self.dynamics.dt*1000,
#             frames=len(self.traj)-1
#         )

#         plt.close()

#         return ani

#     @property
#     def action_space(self):
#         """
#         Defines the action space of the environment using a Box space from OpenAI Gym.

#         The action space is defined based on the lower and upper bounds for the control input specified in the
#         dynamics model. This allows for a continuous range of actions that can be applied to the cartpole system.

#         Returns:
#         - gym.spaces.Box: The action space as a Box object, with low and high bounds derived from the dynamics model's
#         control input bounds.
#         """
#         return gym.spaces.Box(low=self.dynamics.u_lb.numpy(), high=self.dynamics.u_ub.numpy())

#     @property
#     def observation_space(self):
#         """
#         Defines the observation space of the environment using a Box space from OpenAI Gym.

#         The observation space is defined with no bounds on the values, representing the position and velocity of the
#         cart and the angle and angular velocity of the pole. This space allows for any real-valued vector of
#         positions and velocities to be a valid observation in the environment.

#         Returns:
#         - gym.spaces.Box: The observation space as a Box object, with low and high bounds set to negative and
#         positive infinity, respectively, for each dimension of the state vector.
#         """
#         return gym.spaces.Box(
#             low= np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([np.inf,   np.inf,  np.inf,  np.inf])
#         )

class CartPoleEnvNew(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```

    ## Version History
    * v1: `max_time_steps` raised to 500.
        - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.reward_fn = self.compute_reward

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            if self._sutton_barto_reward:
                reward = 0.0
            elif not self._sutton_barto_reward:
                reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            if self._sutton_barto_reward:
                reward = -1.0
            else:
                reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            if self._sutton_barto_reward:
                reward = -1.0
            else:
                reward = 0.0

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), self.reward_fn(self, np.array(self.state, dtype=np.float32), np.array(action, dtype = np.float32)), terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def compute_reward(self, q, u):
        """
        Calculates the reward for given states and control inputs.

        Parameters:
        - q (np.ndarray): The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
        
        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | Cart Position         | -4.8                | 4.8               |
        | 1   | Cart Velocity         | -Inf                | Inf               |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
        | 3   | Pole Angular Velocity | -Inf                | Inf               |

        **Note:** While the ranges above denote the possible values for observation space of each element,
            it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
        -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
        if the cart leaves the `(-2.4, 2.4)` range.
        -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
        if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

        - u (np.ndarray):  The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
        of the fixed force the cart is pushed with.

        - 0: Push cart to the left
        - 1: Push cart to the right

        **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
        the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

        Returns:
        - float: The calculated rewards for the states and inputs. The default reward threshold is 500 for v1 for a time limit of 500 steps.
        """
        

        return ...


def train_cartpole(reward_fn: Callable, pretrained_policy_dict = None):
    env_name = 'CartPole-v1'

    if env_name in registry:
        del registry[env_name]
    register(
        id=env_name,
        entry_point=f'{__name__}:CartPoleEnvNew',
    )

    env = gym.make(env_name)

    # print(env.__dict__)
    # print(env.env)
    # print(env.env.__dict__)
    # print(env.env.env)
    # print(env.env.env.__dict__)

    env.env.env.reward_fn = reward_fn # wtf
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500)
    model.save("ppo_cartpole")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _ = env.step(action)
        env.render()
    return model


if __name__ == "__main__":
    train_cartpole(None)