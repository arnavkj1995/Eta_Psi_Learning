from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.registry import registry

from .recurrent_replay import RecurrentReplayBuffer

registry.register_all(
    BaseReplayBuffer,
    {

        "RecurrentReplayBufferV2": RecurrentReplayBuffer,
    },
)

get_replay = getattr(registry, f"get_{BaseReplayBuffer.type_name()}")