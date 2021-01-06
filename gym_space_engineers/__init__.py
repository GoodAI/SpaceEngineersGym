from gym.envs.registration import register

register(
    id='SpaceEngineers-Dummy-v0',
    entry_point='gym_space_engineers.envs:SpaceEngineersDummyEnv',
)

register(
    id='SpaceEngineers-v0',
    entry_point='gym_space_engineers.envs:SpaceEngineersEnv',
)