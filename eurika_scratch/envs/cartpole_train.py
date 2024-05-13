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
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Tuple, Union
import math
import cv2
from typing import Optional, Tuple, Union
import os

import logging

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

class EpisodeStatsCallback(BaseCallback):
    gamma_val = 0.999
    def __init__(self, verbose=0):
        super(EpisodeStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.total_rewards = 0
        self.episode_lengths = []
        self.current_episode_length = 0
        self.entropy_loss = []
        self.loss = []
        self.policy_gradient_loss = []
        self.value_loss = []



    def _on_step(self) -> bool:
        # Add reward to total for current episode
        self.total_rewards += self.locals["rewards"][0]  # Assuming single environment
        # Increment current episode length
        self.current_episode_length += 1
        # Check if the episode is done
        if self.locals["dones"][0]:
            # Calculate mean reward for the current episode
            mean_reward = self.total_rewards / self.current_episode_length
            self.episode_rewards.append(mean_reward)
            # Record the length of the episode
            self.episode_lengths.append((1 - (self.gamma_val ** self.current_episode_length))/(1 - self.gamma_val))

            # Reset the counters for the next episode
            self.total_rewards = 0
            self.current_episode_length = 0

        return True

    def _on_rollout_end(self):
        # Check for unaccounted rewards and lengths if rollout ends abruptly
        if self.current_episode_length > 0:
            mean_reward = self.total_rewards / self.current_episode_length
            self.episode_rewards.append(mean_reward)
            self.episode_lengths.append(self.current_episode_length)
            # Reset the counters
            self.total_rewards = 0
            self.current_episode_length = 0

    def get_episode_stats(self):
        stats_dict = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "entropy_loss": self.entropy_loss,
            "loss": self.loss,
            "policy_gradient_loss": self.policy_gradient_loss,
            "value_loss": self.value_loss
        }
        return stats_dict

    def dump(self, file_name = "eval_metrics"):
        import json
        with open(file_name) as f:
            f.write(json.dumps(self.get_episode_stats()))

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
        # self.render_frames = []
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
        # print(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
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


def train_cartpole(cfg, reward_fn: Callable, pretrained_model=None, index=-1, eureka_base = False):
    """""""""
    Takes in ... as arguments

    Returns

    model, mean_rewards, episode_length

    """
    env_name = 'CartPole-v1'

    if env_name in registry:
        del registry[env_name]
    register(
        id=env_name,
        entry_point=f'{__name__}:CartPoleEnvNew',
    )

    env = gym.make(env_name)

    if index == -1:
        logging.warn("Index not set")

    # print(env.__dict__)
    # print(env.env)
    # print(env.env.__dict__)
    # print(env.env.env)
    # print(env.env.env.__dict__)

    env.env.env.reward_fn = reward_fn # wtf

    if pretrained_model is None:
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = pretrained_model

    stats_callback = EpisodeStatsCallback()
    model.learn(total_timesteps=cfg.tot_timesteps, callback=stats_callback)

    # Access the lengths of episodes collected during the training. We use these as our fitness function, which we want to maximize
    episode_stats = stats_callback.get_episode_stats()
    model.save("ppo_cartpole")

    obs, _ = env.reset()
    done = False
    frames = []
    
    steps = 0
    while not done and steps < 1000:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _ = env.step(action)
        steps += 1
        frame = env.render()
        frames.append(frame)
    
    # print(frames[0])
    # SAVE FRAMES TO A VIDEO FILE HERE
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(f'cartpole_output_{index}.avi', fourcc, 20.0, (600, 400))

    # Loop over each image and write to the video file
    for frame in frames:
        video.write(frame)

    # Release the video writer
    video.release()
    print(episode_stats["episode_lengths"])
    return model, episode_stats["episode_lengths"], episode_stats["episode_rewards"], [steps]


if __name__ == "__main__":
    fitness_function = train_cartpole(None)
