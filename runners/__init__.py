import hive
from hive.runners.base import Runner

from .single_with_eval import SingleAgentRunnerEval

hive.registry.register('SingleAgentRunnerEval', SingleAgentRunnerEval, Runner)
