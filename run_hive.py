import sys
from pathlib import Path

import torch
import hive
import envs
import nets
import agents
import replays
import runners

from hive.utils.registry import registry
from hive.agents.qnets.base import FunctionApproximator
from hive.main import main

registry.register('GRUCell', torch.nn.GRUCell, FunctionApproximator)

if __name__ == "__main__":
    main()