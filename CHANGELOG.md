# Release 0.0.6
- Added intra episode task randomization
- Scaled reward for the turning task

# Release 0.0.5
- New convention for end effector z

# Release 0.0.4
- Added end effector velocity (with `SE-WalkingSymmetric-v0`)

# Release 0.0.3
- Added turning task
- Added new type of symmetry "per_leg" (only control two legs and then copy/mirror)
- Reduce y_max in the action space
- Added task randomization
- Updated `allowed_leg_angle`
- Reduced max timesteps to 200 (20s)
- Added `v1` env without timeout
- Added `change_task()` method

# Release 0.0.2

- Changed `WalkingRobotIKEnv` to include command as input
- Reduced default max speed in `WalkingRobotIKEnv` (from 15 to 8)
- Added velocity (angular and linear) in `WalkingRobotIKEnv` observation
- Added `SE_SERVER_ADDR` env variable
- Added `symmetric_control` and `allowed_leg_angle` to limit action space
- Longer episodes and limited at 10Hz
- Reduced action space

# Release 0.0.1

- Initial version
