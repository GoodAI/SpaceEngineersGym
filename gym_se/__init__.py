from gym.envs.registration import register

register(
    id='se-dummy-v0',
    entry_point='gym_se.envs:SEDummyEnv',
)

register(
    id='se-v0',
    entry_point='gym_se.envs:SEEnv',
)