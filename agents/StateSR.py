import re
import copy
import wandb
import numpy as np


import hive
import torch
import torch.nn.functional as F

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.utils.utils import LossFn, OptimizerFn, seeder
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import SwitchSchedule, PeriodicSchedule, Schedule

import matplotlib.pyplot as plt
import matplotlib as mpl 

np.random.seed()

class StateSRAgent(Agent):
    def __init__(self, 
                 observation_space,
                 action_space,
                 stack_size,
                 sr_net: FunctionApproximator,
                 optimizer: OptimizerFn,
                 replay_buffer: BaseReplayBuffer,
                 sr_loss_fn: LossFn = torch.nn.MSELoss,
                 target_net_soft_update: bool = False,
                 target_net_update_fraction: float = 0.05,
                 target_net_update_schedule: Schedule = None,
                 batch_size: int = 16,
                 batch_length: int = 25,
                 min_replay_history: int = 1000,
                 min_policy_tuning: int = 200,
                 plot_frequency: int = 10,
                 log_frequency: int = 10,
                 n_step: int = 1,
                 policy: str = "random",
                 id=0,
                 gamma: float = 0.9,
                 epsilon: float = 0.00,
                 grad_clip: float = None,
                 logger=None,
                 device='cuda:0'
                ):

        super().__init__(observation_space, action_space)

        self._gamma = gamma
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._batch_length = batch_length
        self._policy = policy
        self._id = id
        self._n_step = n_step
        self._grad_clip = grad_clip
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        
        self.obs_input_space = observation_space['input']
        self.obs_output_space = observation_space['output']

        self.num_actions = action_space.n
        
        self._replay_buffer = replay_buffer(
            observation_shape=self.obs_input_space.shape,
            observation_dtype=self.obs_input_space.dtype,
            action_shape=action_space.shape,
            action_dtype=action_space.dtype,
            extra_storage_types={"obs_output": (self.obs_output_space.dtype, self.obs_output_space.shape)}
        )
        
        self.param_list = []

        self._state_dim = self.obs_input_space.shape[0]
        
        self._net = sr_net(num_states=self.obs_output_space.shape[0], num_actions=self.num_actions, state_dim=self._state_dim, device=self._device)
        self._hidden_dim = self._net.hidden_dim

        self._target_net_soft_update = target_net_soft_update
        
        if self._target_net_soft_update:
            self._target_net_update_fraction = target_net_update_fraction
            self._target_net = copy.deepcopy(self._net).requires_grad_(False)
        
            if target_net_update_schedule is None:
                self._target_net_update_schedule = PeriodicSchedule(False, True, 1000)
            else:
                self._target_net_update_schedule = target_net_update_schedule()

        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)
        self._policy_schedule = SwitchSchedule(False, True, min_policy_tuning)
        self._plot_schedule = PeriodicSchedule(False, True, plot_frequency)
        
        self.param_list = self.param_list + list(self._net.parameters())
        self._optimizer = optimizer(self.param_list)
        self._sr_loss_fn = sr_loss_fn()
        
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self._id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )

    def _init_log(self):
        return {
            "h": torch.zeros(1, self._hidden_dim).to(device=self._device),
            "phi": torch.zeros(1, self.obs_output_space.shape[0]).to(device=self._device) + 1e-4,
            "len": 0,
            "vis": np.zeros((self.obs_output_space.shape[0]))
        }

    def _get_target_params(self, target_params, curr_params):
        for key in list(target_params.keys()):
            target_params[key] = (
                1 - self._target_net_update_fraction
            ) * target_params[
                key
            ] + self._target_net_update_fraction * curr_params[
                key
            ]
        return target_params

    def _update_target(self):
        """Update the target network."""
        if self._target_net_soft_update:
            self._target_net.load_state_dict(self._get_target_params(self._target_net.state_dict(), self._net.state_dict()))
        else:
            self._target_net.load_state_dict(self._net.state_dict())
            
    def _q_val(self, phi, psi):
        dist = (torch.clamp(phi.unsqueeze(-2), min=1e-8) + torch.clamp(psi, min=1e-8)) / 2.0
        dist = torch.nn.functional.normalize(dist, dim=-1, p=1)
        return -torch.sum(dist * torch.log(dist), dim=-1)

    def _phi(self, seq):
        phi = torch.zeros(len(seq), self._batch_size, self.obs_output_space.shape[0]).to(self._device) + 1e-6
        for i, s in enumerate(seq):
            if i == 0:
                phi[i] = s
            else:
                phi[i] = s + self._gamma * phi[i - 1]
        return phi
    
    def _psi_n_step(self, seq):
        n_step_psi = torch.zeros(1, self._batch_size, self.obs_output_space.shape[0]).to(self._device)
        N = self._n_step
        for i in range(self._n_step):
            if i == 0:
                n_step_psi[N - i - 1] = seq[N - i - 1]
            else:
                n_step_psi[N - i - 1] = seq[N - i - 1] + self._gamma * n_step_psi[N - i]
        return n_step_psi

    def preprocess_update_info(self, update_info):
        """Preprocess the update info."""
        update_info["obs_output"] = update_info["observation"]['output']
        update_info["observation"] = update_info["observation"]['input']

        update_info.pop('next_observation', None)

        return update_info

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffer.

        Args:
            batch: Batch sampled from the replay buffer for the current update.

        Returns:
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """

        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)

        state = batch["observation"].to(torch.float32)
        action = torch.nn.functional.one_hot(batch["action"].to(torch.int64), num_classes=self.num_actions).float()
        reward = batch["reward"].to(torch.float32)
        done = batch["terminated"].to(torch.float32)
        state_onehot = batch["obs_output"].to(torch.float32)

        # State -> [B, S, N], where N is the size of grid
        # Action -> [B, S, M], where M is the number of actions
        state = state.transpose(0, 1)
        action = action.transpose(0, 1)
        reward = reward.transpose(0, 1)
        done = done.transpose(0, 1)
        state_onehot = state_onehot.transpose(0, 1)

        return state, action, reward, done, state_onehot

    def normalize_state(self, state):
        state = F.normalize(state, p=1.0, dim=-1)
        return state

    @torch.no_grad()
    def act(self, observation, episode_log=None):
        if episode_log is None:
            episode_log = self._init_log()

        obs_output = observation['output']
        obs_input = observation['input']
        episode_log["vis"] += obs_output
        episode_log["len"] += 1

        obs_input = torch.tensor(obs_input, device=self._device).unsqueeze(0).to(torch.float32)
        obs_output = torch.tensor(obs_output, device=self._device).unsqueeze(0).to(torch.float32)

        h = episode_log["h"]
        psi, h = self._net.step(obs_input, h)
        
        phi = obs_output + self._gamma * episode_log["phi"]
        
        if (np.random.rand() < self._epsilon and self._training) or not self._policy_schedule.get_value() or self._policy == "random":
            action = np.random.choice(np.arange(self.num_actions))
        elif self._policy == "exploration":
            Q = self._q_val(phi, psi)
            action = np.argmax(Q.cpu().detach().numpy(), axis=-1)[0]
        else:
            raise ValueError("Policy not recognized")
        
        episode_log["h"] = h
        episode_log["phi"] = phi
        
        return action, episode_log

    def update(self, update_info, agent_state=None):
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        if not self._learn_schedule.update():
            return agent_state

        losses = {}
        bl = np.random.choice(np.arange(3, self._batch_length))
        batch = self._replay_buffer.sample(batch_size=self._batch_size, length=bl)
        state, action, _, done, state_onehot = self.preprocess_update_batch(batch)
        
        psi, _ = self._net((state[:-1],))

        if self._target_net_soft_update:
            target_psi, _ = self._target_net((state,))
        else:
            target_psi, _ = self._net((state,))
        
        # Here we assume that \psi_{t}(a_t) = s_{t+1} + \gamma * \phi_{t+1}(a_{t+1}')
        if not self._policy_schedule.update() or self._policy == 'random':
            target_action_prob = torch.ones(*target_psi.shape[:-1]) / self.num_actions
        else:
            phi = self._phi(state_onehot)
            Q_val = self._q_val(phi[-1:], target_psi)
            target_action_prob = torch.nn.functional.one_hot(torch.argmax(Q_val, dim=-1), num_classes=self.num_actions).float()
            
        # Compute the SR loss function
        # sr_pred -> [2, B, A, S]
        sr_pred = torch.einsum('tbai,tba->tbi', psi, action[-self._n_step - 1:-self._n_step])
        sr_pred_target = torch.einsum('tbai,tba->tbi', target_psi, target_action_prob.to(self._device))
   
        # Compute TD error for SR
        n_step_state = self._psi_n_step(state_onehot[-self._n_step - 1: -1])[0:1]
        sr_target =  n_step_state + (1 - done[-1].unsqueeze(-1)) * np.power(self._gamma, self._n_step) * sr_pred_target

        losses['sr'] = self._sr_loss_fn(sr_pred, sr_target.detach())
            
        total_loss = torch.sum(torch.stack(list(losses.values())))

        self._optimizer.zero_grad()
        total_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.param_list, self._grad_clip
            )
        self._optimizer.step()

        if self._logger.update_step(self._timescale):
            self._logger.log_scalar("SR Loss", losses['sr'].item(), self._timescale)
            
        # Update target network
        if self._target_net_soft_update and (not self._policy_schedule.get_value() or self._target_net_update_schedule.update()):
            self._update_target()
        
        return agent_state

    def get_policy(self):
        policy = np.zeros(self.num_states)
        for i in range(self.num_states):
            policy[i] = np.argmax(self.q[:,i])
        return policy
    
    def save(self, foldername):
        pass
    
    def load(self, foldername):
        pass
