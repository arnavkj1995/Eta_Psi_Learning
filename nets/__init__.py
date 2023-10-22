import hive
from hive.agents.qnets.base import FunctionApproximator

from .mlp import MLP
from .rec import RecurrentEncoderNetwork
from .rec import RecurrentSRNetwork

hive.registry.register('MLP', MLP, FunctionApproximator)
hive.registry.register('RecurrentSR', RecurrentSRNetwork, FunctionApproximator)
hive.registry.register('RecurrentEncoder', RecurrentEncoderNetwork, FunctionApproximator)
