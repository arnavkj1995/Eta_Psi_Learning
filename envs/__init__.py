from gymnasium.envs.registration import register

register(
    'GridWorld-v0',
    entry_point='envs.grids:GridWorld'
)

register(
    'TwoRooms-v0',
    entry_point='envs.grids:TwoRooms'
)

register(
    'FourRooms-v0',
    entry_point='envs.grids:FourRooms'
)

register(
    'ChainEnv-v0',
    entry_point='envs.chains:ChainEnv'
)

register(
    'RiverSwim-v0',
    entry_point='envs.chains:RiverSwimEnv'
)

register(
    'DiscreteReacher-v0',
    entry_point='envs.control:Reacher'
)

register(
    'DiscretePusher-v0',
    entry_point='envs.control:Pusher'
)
