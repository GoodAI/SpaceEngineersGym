# Release 0.0.3
- Added turning task
- Added new type of symmetry "per_leg" (only control two legs and then copy/mirror)
- Reduce y_max in the action space

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
