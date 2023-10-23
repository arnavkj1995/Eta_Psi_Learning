import re
import copy
import wandb
import numpy as np


import hive
import torch
import torch.nn.functional as F
from typing import Tuple, Union

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.utils.utils import LossFn, OptimizerFn, seeder
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import SwitchSchedule, PeriodicSchedule, Schedule
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)

import matplotlib.pyplot as plt
import matplotlib as mpl 

np.random.seed()

# Similar to the the critic implementation of TD3
class TD3PsiNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_dim: int,
        output_dim: int,
        psi_net: FunctionApproximator,
        n_psi_nets: int,
        action_shape: Tuple[int],
    ) -> None:
        super().__init__()
        self._n_psi_nets = n_psi_nets
        self._psi_nets = torch.nn.ModuleList([psi_net(in_dim=int(representation_dim + np.prod(action_shape)), out_dim=output_dim) for _ in range(n_psi_nets)])

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return [psi_net(x).unsqueeze(1) for psi_net in self._psi_nets]

    def to(self, device):
        for psi_net in self._psi_nets:
            psi_net.to(device)

    def psi1(self, obs, actions):
        """Returns the value according to only the first network."""
        x = torch.cat([obs, actions], dim=-1)
        return self._psi_nets[0](x)

class TD3ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_dim: int,
        actor_net: FunctionApproximator,
        action_shape: Tuple[int],
        use_tanh=True,
    ) -> None:
        
        super().__init__()

        self._action_shape = action_shape
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(in_dim=representation_dim, out_dim=action_shape[0])

        actor_modules = [actor_network]
        if use_tanh:
            actor_modules.append(torch.nn.Tanh())
        self.actor = torch.nn.Sequential(*actor_modules)

    def forward(self, x):
        x = self.actor(x)
        return torch.reshape(x, (x.size(0), *self._action_shape))

class StateSRPGAgent(Agent):
    def __init__(self, 
                 observation_space,
                 action_space,
                 stack_size,
                 traj_encoder_net: FunctionApproximator,
                 psi_net: FunctionApproximator,
                 actor_net: FunctionApproximator,
                 actor_optimizer_fn: OptimizerFn,
                 psi_optimizer_fn: OptimizerFn,
                 replay_buffer: BaseReplayBuffer,
                 psi_loss_fn: LossFn = torch.nn.MSELoss,
                 init_fn: InitializationFn = None,
                 target_net_update_fraction: float = 0.005,
                 n_psi_nets: int = 2,
                 grad_clip: float = None,
                 batch_size: int = 64,
                 batch_length: int = 25,
                 min_replay_history: int = 1000,
                 min_policy_tuning: int = 1000,
                 plot_frequency: int = 10,
                 log_frequency: int = 10,
                 n_step: int = 1,
                 policy: str = "random",
                 id=0,
                 gamma: float = 0.9,
                 epsilon: float = 0.00,
                 logger=None,
                 bin_size=11,
                 num_bins=2,
                 device='cuda:0',
                 action_noise: float = 0,
                 target_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 policy_update_frequency: int = 2,
                 update_frequency: int=1
                ):

        super().__init__(observation_space, action_space)

        self._gamma = gamma
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._batch_length = batch_length
        self._policy = policy
        self._id = id
        self._n_step = n_step
        self._bin_size = bin_size
        self._num_bins = num_bins
        self._n_psi_nets = n_psi_nets
        self._grad_clip = grad_clip
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        
        self.obs_input_space = observation_space['input']
        self.obs_output_space = observation_space['output']

        self.num_actions = action_space.shape[0]
        self._action_min = self._action_space.low
        self._action_max = self._action_space.high
        self._action_scaling = 0.5 * (self._action_max - self._action_min)
        self._scale_actions = np.isfinite(self._action_scaling).all()
        self._action_min_tensor = torch.as_tensor(self._action_min, device=self._device)
        self._action_max_tensor = torch.as_tensor(self._action_max, device=self._device)
        self._init_fn = create_init_weights_fn(init_fn)
        self._action_noise = action_noise
        self._target_noise = target_noise
        self._target_noise_clip = target_noise_clip

        self._replay_buffer = replay_buffer(
            observation_shape=self.obs_input_space.shape,
            observation_dtype=self.obs_input_space.dtype,
            action_shape=action_space.shape,
            action_dtype=action_space.dtype,
            extra_storage_types={"obs_output": (self.obs_output_space.dtype, self.obs_output_space.shape)}
        )
        
        self._traj_encoder = traj_encoder_net(state_dim=self.obs_input_space.shape[0], device=self._device)
        self._hidden_dim = self._traj_encoder.hidden_dim

        self._psi_net = TD3PsiNetwork(self._hidden_dim, self.obs_output_space.shape[0], psi_net, self._n_psi_nets, action_space.shape)        
        self._actor = TD3ActorNetwork(self._hidden_dim, actor_net, action_space.shape, self._scale_actions)

        self._traj_encoder.to(self._device)
        self._actor.to(self._device)
        self._psi_net.to(self._device)

        self._actor.apply(self._init_fn)
        self._psi_net.apply(self._init_fn)
        self._traj_encoder.apply(self._init_fn)

        self._target_actor = copy.deepcopy(self._actor).requires_grad_(False)
        self._target_psi_net = copy.deepcopy(self._psi_net).requires_grad_(False)
        self._target_traj_encoder = copy.deepcopy(self._traj_encoder).requires_grad_(False)
        self._target_net_update_fraction = target_net_update_fraction

        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)
        self._policy_update_schedule = PeriodicSchedule(
            False, True, policy_update_frequency
        )
        self._update_schedule = PeriodicSchedule(False, True, update_frequency)
        self._plot_schedule = PeriodicSchedule(False, True, plot_frequency)
        self._pretrain_psi_schedule = SwitchSchedule(False, True, min_policy_tuning)
        
        if psi_optimizer_fn is None:
            psi_optimizer_fn = torch.optim.Adam

        if actor_optimizer_fn is None:
            actor_optimizer_fn = torch.optim.Adam
        
        self._psi_optimizer = psi_optimizer_fn(list(self._psi_net.parameters()) + list(self._traj_encoder.parameters()))
        self._actor_optimizer = actor_optimizer_fn(list(self._actor.parameters()))
        
        self._psi_loss_fn = psi_loss_fn()

        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self._id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )

        self._training = False

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor.train()
        self._psi_net.train()
        self._traj_encoder.train()
        self._target_actor.train()
        self._target_psi_net.train()
        self._target_traj_encoder.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor.eval()
        self._psi_net.eval()
        self._traj_encoder.eval()
        self._target_actor.eval()
        self._target_psi_net.eval()
        self._target_traj_encoder.eval()

    def _init_log(self):
        return {
            "h": torch.zeros(1, self._hidden_dim).to(device=self._device),
            "phi": torch.zeros(1, self.obs_output_space.shape[0]).to(device=self._device) + 1e-4,
            "len": 0,
            "vis": np.zeros((self.obs_output_space.shape[0])),
        }

    def scale_action(self, actions):
        """Scales actions to [-1, 1]."""
        if self._scale_actions:
            return ((actions - self._action_min) / self._action_scaling) - 1.0
        else:
            return actions

    def unscale_actions(self, actions):
        """Unscales actions from [-1, 1] to expected scale."""
        if self._scale_actions:
            return ((actions + 1.0) * self._action_scaling) + self._action_min
        else:
            return actions

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
        for network, target_network in [
            (self._actor, self._target_actor),
            (self._psi_net, self._target_psi_net),
            (self._traj_encoder, self._target_traj_encoder),
        ]:
            target_params = target_network.state_dict()
            current_params = network.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (1 - self._target_net_update_fraction) * target_params[
                    key
                ] + self._target_net_update_fraction * current_params[key]
            target_network.load_state_dict(target_params)
            
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
    
    def _td_target(self, state, bootstrap, done):
        return state + np.power(self._gamma, self._n_step) * (1 - done.unsqueeze(-1)) * bootstrap

    def preprocess_update_info(self, update_info):
        """Preprocess the update info."""
        if update_info["truncated"] == True:
            self._epsilon = max(0.00, self._epsilon * 0.99)
        update_info["obs_output"] = update_info["observation"]['output']
        update_info["observation"] = update_info["observation"]['input']
        update_info["action"] = self.scale_action(update_info["action"])

        update_info.pop('next_observation', None)
        
        return update_info

    def preprocess_update_batch(self, batch):
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        

        # FIXME: Hardcoded
        state = batch["observation"].to(torch.float32)
        action = batch["action"].to(torch.float32)
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
        h = self._traj_encoder.step(obs_input, h)
        action = self._actor(h)
        
        phi = obs_output + self._gamma * episode_log["phi"]
        if self._training:
            noise = torch.randn_like(action, requires_grad=False) * self._action_noise
            action = action + noise
        
        action = action.cpu().detach().numpy()[0]
        if self._scale_actions:
            action = self.unscale_actions(action)

        action = np.clip(action, self._action_min, self._action_max)

        if not self._learn_schedule.get_value():
            action = np.random.uniform(self._action_min, self._action_max)

        episode_log["h"] = h
        episode_log["phi"] = phi
        return action, episode_log

    def update(self, update_info, agent_state=None):
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        if not self._learn_schedule.update():
            return agent_state

        # Extract batch from Replay Buffer
        bl = np.random.choice(np.arange(3, self._batch_length - self._n_step))  + self._n_step
        batch = self._replay_buffer.sample(batch_size=self._batch_size, length=bl)
        state, action, _, done, state_onehot = self.preprocess_update_batch(batch)

        with torch.no_grad():
            noise = torch.randn_like(action[0], requires_grad=False) * self._target_noise
            noise = torch.clamp(
                noise, -self._target_noise_clip, self._target_noise_clip
            )
            
            encoded_traj = self._target_traj_encoder([state])[-1]
            next_actions = self._target_actor(encoded_traj) + noise
            if self._scale_actions:
                next_actions = torch.clamp(next_actions, -1, 1)
            else:
                next_actions = torch.clamp(
                    next_actions, self._action_min_tensor, self._action_max_tensor
                )

            next_psi_vals = torch.cat(self._target_psi_net(encoded_traj, next_actions), dim=1)
            phi = self._phi(state_onehot)[-1]

            next_q_vals = self._q_val(phi, next_psi_vals)

            ##### Get PHI values
            min_q_inds = torch.argmin(next_q_vals, dim=-1, keepdim=False)
            mask = torch.nn.functional.one_hot(min_q_inds, num_classes=self._n_psi_nets).float()
            min_psi_vals = torch.einsum('bai,ba->bi', next_psi_vals, mask)

            target_psi_vals = (
                state_onehot[-1] + (1 - done[-1].unsqueeze(-1)) * self._gamma * min_psi_vals
            ).unsqueeze(1).detach()
        
        # Encode State -> Get Psi using current action -> Update all psi_nets
        encoded_state = self._traj_encoder([state])[-2]
        psi_vals = self._psi_net(encoded_state, action[-2])
        psi_loss = sum(
                [self._psi_loss_fn(psi_, target_psi_vals) for psi_ in psi_vals]
            )
        self._psi_optimizer.zero_grad()
        psi_loss.backward()

        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self._psi_net.parameters(), self._grad_clip
            )
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self._traj_encoder.parameters(), self._grad_clip
            )
        self._psi_optimizer.step()

        if not self._pretrain_psi_schedule.update():
            actor_loss = torch.tensor([0.0])

        else:
            
            # Update policy with policy delay
            if self._policy_update_schedule.update():
                
                encoded_state = self._traj_encoder([state])[-2].detach()
                psi_actor = self._psi_net.psi1(
                        encoded_state, self._actor(encoded_state)
                    )
                
                actor_dist = torch.clamp(self._phi(state_onehot)[-2], min=1e-8) + torch.clamp(psi_actor, min=1e-8)
                actor_dist = torch.nn.functional.normalize(actor_dist, dim=-1, p=1)

                pg_coeff = torch.log(actor_dist) + 1
                psi_actor_updated = pg_coeff.detach() * psi_actor * ((1. - self._gamma))
                actor_loss = torch.mean(torch.sum(psi_actor_updated, axis=-1))
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._actor.parameters(), self._grad_clip
                    )
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._traj_encoder.parameters(), self._grad_clip
                    )
                self._actor_optimizer.step()
                self._update_target()
                if self._logger.should_log(self._timescale):
                    self._logger.log_scalar("actor_loss", actor_loss, self._timescale)
                
        if self._logger.update_step(self._timescale):
            self._logger.log_scalar("SR Loss", psi_loss.item(), self._timescale)
            self._logger.log_scalar("SR_norm", psi_vals[0].sum(dim=-1).mean().item(), self._timescale)
            
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
