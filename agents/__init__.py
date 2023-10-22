import hive
from hive.agents.agent import Agent

from .StateSR import StateSRAgent
from .StateSRPG import StateSRPGAgent

hive.registry.register('StateSRAgent', StateSRAgent, Agent)
hive.registry.register('StateSRPGAgent', StateSRPGAgent, Agent)
