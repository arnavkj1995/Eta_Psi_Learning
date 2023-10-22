import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class GridWorld(gym.Env):
    def __init__(self, 
                width=5, 
                height=5, 
                explore=False,
                seed=None, 
                timeout=200,
                **kwargs):
        self._width = width
        self._height = height
        self._num_actions = 4
        self.timeout = timeout
        self.explore = explore
        
        # 0: left, 1: down, 2: right, 3: up
        self._directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        self._agent_start_pos = [0, 0]

        self._vis_freq = np.zeros((self._width * self._height))

        self._episode_id = 0
        self._obstacles = []

        # Actions and Observation are one-hot embedding
        self.action_space = spaces.Discrete(self._num_actions)
        self.observation_space = gym.spaces.Dict({"input": gym.spaces.Box(low=0, high=1, shape=(self._width * self._height,)), 
                                                  "output": gym.spaces.Box(low=0, high=1, shape=(self._width * self._height,))})

        self.seed(seed)

    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width
    
    @property
    def total_states(self):
        return self._height * self._width
        
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
        
        self._vis_freq = np.zeros((self._width * self._height))
        self._vis_freq[self._agent[0] * self._width + self._agent[1]] = 1
        
        for p in self._obstacles:
            self._vis_freq[p[0] * self._width + p[1]] = 1

        return {'input': self.get_state(), 'output': self.get_state()}, {}

    def set_timeout(self, t):
        self.timeout = t
    
    def set_explore(self, x):
        self.explore = x
        
    def valid_pos(self, pos):
        """Check if position is valid."""
        if pos in self._obstacles:
            return False
        if pos[0] < 0 or pos[0] >= self._height:
            return False
        if pos[1] < 0 or pos[1] >= self._width:
            return False
        return True

    def translate(self, offset):
        """"Translate agent pixel.
        Args:
            offset: (x, y) tuple of offsets.
        """
        new_pos = [p + o for p, o in zip(self._agent, offset)]
        if self.valid_pos(new_pos):
            self._agent = new_pos

    def get_state(self):
        state_obs = np.zeros(self._width * self._height)
        state_obs[self._agent[0] * self._width + self._agent[1]] = 1
        return state_obs

    def step(self, action):
        reward=0
        truncated = False
        
        self.translate(self._directions[action])
        self._vis_freq[self._agent[0] * self._width + self._agent[1]] += 1
        self._step += 1
            
        if self._step == self.timeout:
            truncated=True
        
        obs = self.get_state()
        self._vis_freq = self._vis_freq + obs

        if self.explore and not np.any(self._vis_freq == 0):
            truncated = True

        return {'input': obs, 'output': obs}, reward, False, truncated, {}

class TwoRooms(GridWorld):
    def __init__(self, 
                width=9, 
                height=3, 
                seed=None, 
                timeout=200, 
                **kwargs):
        super().__init__(width=width, 
                         height=height, 
                         seed=seed,
                         timeout=timeout, 
                         **kwargs)

        self._agent_init_pos = [1, 4]
        self._obstacles.append([0, 4])
        self._obstacles.append([2, 4])

class FourRooms(GridWorld):
    def __init__(self, 
                width=7, 
                height=7, 
                seed=None, 
                timeout=200, 
                **kwargs):
        super().__init__(width=width, 
                         height=height, 
                         seed=seed,
                         timeout=timeout, 
                         **kwargs)

        self._obstacles.append([int(self._height / 2), int(self._width / 2)])
        for i in range(int(self._width / 2)):
            if i != int((self._width / 2) / 2):
                self._obstacles.append([int(self._height / 2), i])
                self._obstacles.append([int(self._height / 2), self._width - i - 1])
        
        for i in range(int(self._height / 2)):
            if i != int((self._height / 2) / 2):
                self._obstacles.append([i, int(self._width / 2)])
                self._obstacles.append([self._height - i - 1, int(self._width / 2)])
