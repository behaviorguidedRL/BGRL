import numpy as np
import sys
import math
import gym
from gym.utils import seeding
from gym import spaces


class PointmassEnv(gym.Env):
    def __init__(self,goal_reward=10, actuation_cost_coeff=30,
    	         distance_cost_coeff=1, init_sigma=0.1):
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.init_mu = np.zeros([2])
        self.init_sigma = 0.1
        self.observation_space = spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None
        )
        self.action_space = spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(2,)
        )
        self.env_spec = gym.spec('pointmass-v0')
        self.env_spec.timestep_limit = 100
        self.dt = 0.1

    def reset(self):
    	unclipped_observation = self.init_mu + self.init_sigma * \
    	    np.random.randn(2)
    	o_lb, o_ub = self.observation_space.low, self.observation_space.high
    	self.observation = np.clip(unclipped_observation, o_lb, o_ub)
    	return self.observation

    def get_current_obs(self):
    	return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.observation.copy() + action * self.dt
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        reward = -np.linalg.norm(self.observation - np.ones([2]) * 2.0)

        return next_obs, reward, False, {'pos': next_obs}

class PointmassNonlinearEnv(PointmassEnv):
    def __init__(self, *args, **kwargs):
        PointmassEnv.__init__(self, *args, **kwargs)

    def step(self, action):
        # nonlinear dynamics
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.observation.copy() + action * self.dt
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        reward = -np.linalg.norm(self.observation - np.ones([2]) * 2.0)

        return next_obs, reward, False, {'pos': next_obs}        