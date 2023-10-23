import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
     
class DiscretizedEnv(gym.Env):
    def __init__(self, 
                bins=11, 
                timeout=100, 
                seed=None, 
                **kwargs
                ):
        self.bins = bins
        self._bins = bins
        self.timeout = timeout	
        self._episode_id = 0
        self.timeout = timeout

        self.observation_space = gym.spaces.Dict({"input": self._env.observation_space, 
                                          "output": gym.spaces.Box(low=0, high=1, shape=(self._bins * self._bins,))})
        self.action_space = self._env.action_space
        self.out_state_bins = self.get_state_bins(self.out_state_lows, self.out_state_highs)
        self.seed(seed)

    @property
    def vis_freq(self):
        return self._vis_freq
    
    @property
    def height(self):
        return self._bins
    
    @property
    def width(self):
        return self._bins
    
    @property
    def total_states(self):
        return self._bins * self._bins

    @property
    def agent(self):
        return self._agent
    
    @property
    def obs_output_type(self):
        return self._obs_output_type
    
    @property
    def obs_output_shape(self):
        return self.obs_output_space.shape
    
    def compute_reward(self, r):
        return r
    
    def compute_termination(self, d):
        return d

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self, seed=None):
        self._step = 0
        obs, info = self._env.reset()
        self._agent = obs
        
        output_obs = self.get_state(self.transform_out_state(self._agent), self.out_state_bins)
        
        self._vis_freq = output_obs

        return {'input': obs, 'output': output_obs}, info

    def set_timeout(self, t):
        self.timeout = t
    
    def set_explore(self, x):
        self.explore = x
        
    def discretize_range(self, lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

    def discretize_value(self, value, bins):
        return np.asscalar(np.digitize(x=value, bins=bins, right=True))

    def get_state_bins(self, low_vals, high_vals):
        state_bins = []

        for i, v in enumerate(zip(low_vals, high_vals)):
            state_bins.append(self.discretize_range(v[0], v[1], self._bins))

        return state_bins

    def get_state(self, agent_state, state_bins):
        valid_inds = [0, 1]
        state = np.zeros(np.power(self._bins, len(valid_inds)))
        vec_ind = 0
        for i, ind in enumerate(valid_inds):
            vec_ind +=  np.power(self._bins, i) * self.discretize_value(agent_state[ind], state_bins[ind])
        state[vec_ind] = 1
        return state
        
    def transform_inp_state(self, obs):
        return obs
    
    def transform_out_state(self, obs):
        obs = self.normalize_obs(obs)
        obs = self.proj_func(obs)
        return [obs[i] for i in self._obs_output_dims]

    def render(self):
        return self._env.render()

    def normalize_obs(self, obs):
        return obs

    def step(self, action):
        obs, r, _, _, info = self._env.step(action)

        terminated, truncated = False, False
        self._agent = obs
        
        self._step += 1
        if self._step == self.timeout:	
            truncated=True    

        output_obs = self.get_state(self.transform_out_state(self._agent), self.out_state_bins)
        
        self._vis_freq += output_obs

        return {'input': obs, 'output': output_obs}, r, terminated, truncated, info
        
class Pusher(DiscretizedEnv):
    def __init__(self, 
                 bins=11, 
                 timeout=100, 
                 seed=None, 
                 **kwargs):
        
        self._env = gym.make('Pusher-v4', render_mode='rgb_array')
        
        # Took from the range of x-y coordinates of the fingertip in the Pusher environment
        self.out_state_lows = [-0.3,-0.6, -0.31]
        self.out_state_highs = [0.8, 0.2, 0.55]

        super().__init__(bins=bins,
                         timeout=timeout,
                         seed=seed,
                         **kwargs)
    
    def transform_inp_state(self, obs):
        return obs
    
    def transform_out_state(self, obs):
        return obs[14:16]

class Reacher(DiscretizedEnv):
    def __init__(self, 
                 bins=11, 
                 timeout=100,
                 seed=None, 
                 **kwargs):
    
        self._env = gym.make('Reacher-v4', render_mode='rgb_array')
        self._obs_output_dims = [0, 1]

        self.out_state_lows = [-0.21, -0.21]
        self.out_state_highs = [0.21, 0.21]

        super().__init__(bins=bins,
                         timeout=timeout,
                         seed=seed,
                         **kwargs)
    
    def transform_out_state(self, obs):
        return [obs[8] + obs[4], obs[9] + obs[5]]
