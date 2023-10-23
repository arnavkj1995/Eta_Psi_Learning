from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from hive.agents.qnets.base import FunctionApproximator

class RecurrentSRNetwork(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        state_dim: int,
        encoded_dim: int,
        state_encoder_net: FunctionApproximator = None,
        transition_net: FunctionApproximator = None,
        sr_net: FunctionApproximator = None,
        device: torch.device = torch.device("cpu"),
        last_step: bool = True
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_states = num_states
        self._device = device
        self._last_step = last_step

        if state_encoder_net is not None:
            self._state_encoder = state_encoder_net(in_dim=state_dim, out_dim=encoded_dim)
            self._state_encoder.to(self._device)
        else:
            self._state_encoder = None
            
        self._transition_net = transition_net(input_size=encoded_dim) 
        self.hidden_dim = self._transition_net.hidden_size

        sr_input_dim = self.hidden_dim + encoded_dim
        self._sr_net = sr_net(in_dim=sr_input_dim, out_dim=self._num_actions * self._num_states)

        self._transition_net.to(device=device)
        self._sr_net.to(device=device)

    def recurrent_step(self, i, h):
        return self._transition_net(i, h)

    def psi(self, state):
        return self._sr_net(state)

    def step(self, state, hidden=None):
        if self._state_encoder:
            state = self._state_encoder(state)
        hidden = self.recurrent_step(state, hidden)
        psi = self.psi(torch.cat([state, hidden], dim=-1))
        psi = psi.view(*psi.shape[:-1], self._num_actions, self._num_states)
        return psi, hidden

    def forward(self, x):
        # Encoding state
        # State_ -> [B, S, E]; Action_ -> [B, S, E]
        state = x[0]
        if self._state_encoder:
            state = self._state_encoder(state)

        h_list = []
        h = torch.zeros(state.shape[1], self._transition_net.hidden_size, device=self._device)
        for s in state:
            h = self.recurrent_step(s, h)
            h_list.append(h)

        hidden = torch.stack(h_list, dim=0)

        if self._last_step:
            psi = self.psi(torch.cat([state[-1:], hidden[-1:]], dim=-1))
            psi = psi.view(*psi.shape[:-1], self._num_actions, self._num_states)
        else:
            psi = self.psi(torch.cat([state, hidden], dim=-1))
            psi = psi.view(*psi.shape[:-1], self._num_actions, self._num_states)

        return psi, hidden

class RecurrentEncoderNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        encoded_dim: int,
        state_encoder_net: FunctionApproximator = None,
        transition_net: FunctionApproximator = None,
        device: torch.device = torch.device("cpu"),
        last_step: bool = True
    ):
        super().__init__()
        self._device = device
        self._last_step = last_step

        if state_encoder_net is not None:
            self._state_encoder = state_encoder_net(in_dim=state_dim, out_dim=encoded_dim)
            self._state_encoder.to(self._device)
        else:
            self._state_encoder = None
            
        self._transition_net = transition_net(input_size=encoded_dim) 
        self.hidden_dim = self._transition_net.hidden_size

        self._transition_net.to(device=device)
        
    def recurrent_step(self, i, h):
        return self._transition_net(i, h)

    def step(self, state, hidden=None):
        if self._state_encoder:
            state = self._state_encoder(state)
        hidden = self.recurrent_step(state, hidden)
        return hidden

    def forward(self, x):
        state = x[0]
        if self._state_encoder:
            state = self._state_encoder(state)

        h_list = []
        h = torch.zeros(state.shape[1], self._transition_net.hidden_size, device=self._device)
        for s in state:
            h = self.recurrent_step(s, h)
            h_list.append(h)

        hidden = torch.stack(h_list, dim=0)

        return hidden
