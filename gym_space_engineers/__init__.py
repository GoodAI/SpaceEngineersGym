from gym.envs.registration import register

register(
    id="SpaceEngineers-Dummy-v0",
    entry_point="gym_space_engineers.envs:SpaceEngineersDummyEnv",
)

register(
    id="SpaceEngineers-v0",
    entry_point="gym_space_engineers.envs:SpaceEngineersEnv",
)

register(
    id="SpaceEngineers-RoboticLeg-v0",
    entry_point="gym_space_engineers.envs:RoboticLegEnv",
)

register(
    id="SpaceEngineers-WalkingRobot-IK-v0",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=400,  # around 40s of interaction
    kwargs={"control_frequency": 10},  # 10Hz
)
# Timeout
register(
    id="SpaceEngineers-WalkingRobot-IK-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    kwargs={"control_frequency": 10},  # 10Hz
)
