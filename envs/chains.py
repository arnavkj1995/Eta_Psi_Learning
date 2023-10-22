import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class ChainEnv(gym.Env):
    def __init__(self, 
                 width=5, 
                 start_pos=0, 
                 explore=False, 
                 seed=None, 
                 timeout=200,
                 **kwargs):
        self._width = width
        self._num_actions = 2
        self.timeout = timeout
        self.explore = explore
        
        self._directions = [-1, 1] 
        assert len(self._directions) == self._num_actions

        self._agent_start_pos= start_pos
        self._agent = self._agent_start_pos

        self._vis_freq = np.zeros((self._width))

        self._episode_id = 0

        # Actions and Observation are one-hot embedding
        self.action_space = gym.spaces.Discrete(self._num_actions)
        self.observation_space = gym.spaces.Dict({"input": gym.spaces.Box(low=0, high=1, shape=(self._width,)), 
                                                  "output": gym.spaces.Box(low=0, high=1, shape=(self._width,))})

        self.seed(seed)

    @property
    def height(self):
        return 1
    
    @property
    def width(self):
        return self._width
    
    @property
    def total_states(self):
        return self._width

    @property
    def vis_freq(self):
        return self._vis_freq
        
    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_agent(self, pos):
        self._agent = pos
        
    def reset(self, **kwargs):
        # FIXME: Agent hardcoded to the start state
        self._agent = self._agent_start_pos

        self._step = 0
        self._episode_id += 1
        
        self._vis_freq = np.zeros((self._width))
        self._vis_freq[self._agent] = 1
        
        return {'input': self.get_state(), 'output': self.get_state()}, {}

    def valid_pos(self, pos):
        """Check if position is valid."""
        if pos < 0 or pos >= self._width:
            return False
        
        return True

    def translate(self, offset):
        """"Translate agent pixel.
        Args:
            offset: (x, y) tuple of offsets.
        """
        new_pos = self._agent + offset
        if self.valid_pos(new_pos):
            self._agent = new_pos

    def get_state(self, obs_type='onehot'):
        state_obs = np.zeros(self._width)
        state_obs[self._agent] = 1
        return state_obs

    def set_timeout(self, t):
        self.timeout = t
    
    def set_explore(self, x):
        self.explore = x

    def step(self, action):
        reward=0
        truncated = False
        
        self.translate(self._directions[action])
            
        self._step += 1
        if self._step == self.timeout:	
            truncated=True

        obs = self.get_state()
        self._vis_freq = self._vis_freq + obs

        if self.explore and not np.any(self._vis_freq == 0):
            truncated = True

        return {'input': obs, 'output': obs}, reward, False, truncated, {}

class RiverSwimEnv(ChainEnv):
    def __init__(self, 
                 width=5, 
                 start_pos=0, 
                 explore=False, 
                 seed=None, 
                 timeout=200, 
                 **kwargs):
        super().__init__(width=width,
                         start_pos=start_pos,
                         explore=explore,
                         seed=seed,
                         timeout=timeout,
                         **kwargs)
    
    def step(self, action):
        reward=0
        terminated, truncated = False, False

        if action == 1:
            r = np.random.rand()
            if self._agent == self.width - 1:
                if r > 0.3:
                    action = 0
            else:
                if r < 0.1:
                    action = 0
                elif r < 0.7:
                    action = -1

        if action != -1:           
            self.translate(self._directions[action])
        
        self._step += 1
        if self._step == self.timeout:	
            truncated = terminated = True

        obs = self.get_state()
        self._vis_freq = self._vis_freq + obs

        if self.explore and not np.any(self._vis_freq == 0):
            truncated = terminated = True

        return {'input': obs, 'output': obs}, reward, terminated, truncated, {}
        
