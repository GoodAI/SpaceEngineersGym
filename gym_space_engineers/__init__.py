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
    id="SpaceEngineers-WalkingRobot-IK-v2",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={"control_frequency": 10},  # 10Hz
)

register(
    id="SE-WalkingTest-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    # max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": False,
        "add_end_effector_velocity": True,
    },
)

register(
    id="SE-Forward-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": False,
        "add_end_effector_velocity": True,
    },
)

register(
    id="SE-Symmetric-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": True,
        "add_end_effector_velocity": True,
    },
)

register(
    id="SE-TurnLeft-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": False,
        "add_end_effector_velocity": True,
        "task": "turn_left",
    },
)

register(
    id="SE-Generic-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": False,
        "add_end_effector_velocity": True,
        "task": "generic_locomotion",
        "weight_distance_traveled": 1.0,
        "weight_heading_deviation": 20.0,
        "desired_angular_speed": 10.0,
        "desired_linear_speed": 4.0,
    },
)


register(
    id="SE-MultiTask-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,  # around 20s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "symmetric_control": False,
        "add_end_effector_velocity": True,
        "task": "forward",
        "randomize_task": True,
        "randomize_interval": 80,  # change task every 8s
    },
)

# No Timeout
register(
    id="SpaceEngineers-WalkingRobot-IK-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    kwargs={"control_frequency": 10},  # 10Hz
)

register(
    id="SpaceEngineers-RoboticLeg-v1",
    entry_point="gym_space_engineers.envs:RoboticLegEnvV1",
)

register(
    id="SpaceEngineers-WalkingRobot-OpenLoop-v0",
    entry_point="gym_space_engineers.envs:WalkingRobotOpenLoopEnv",
)

corrections_kwargs = {
    "add_end_effector_velocity": True,
    "max_action": 3.0,
    "correction_only": True,
    "weight_center_deviation": 3,
    "include_phase": True,
    "use_terrain_sensors": True,
    "disable_early_termination": True,
    "max_speed": 100.0,
    "friction": 100.0,
}

register(
    id="SE-Corrections-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,
    kwargs={
        **corrections_kwargs,
    },
)

register(
    id="SE-Corrections-TurnLeft-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,
    kwargs={
        **corrections_kwargs,
        "task": "turn_left",
    },
)

register(
    id="SE-Corrections-Multi-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    max_episode_steps=200,
    kwargs={
        **corrections_kwargs,
        "randomize_task": True,
        "randomize_interval": 80,
    },
)

register(
    id="SE-Corrections-Control-v1",
    entry_point="gym_space_engineers.envs:WalkingRobotIKEnv",
    kwargs={
        **corrections_kwargs,
        "disable_early_termination": False,
    },
)