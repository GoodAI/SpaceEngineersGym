# Walking robot gym

## How to load world

- Make sure that you are connected to Steam
- Run *VeriDream/StartWalkingRobots.bat*
- There will be a popup from Steam where you have to confirm that you want to run Space Engineers with some additional parameters
- The game should load a world with a platform for the robot to walk on

## Train an agent

RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo/tree/misc/veridream

Walk forward (with the quadruped):
```
python train.py --algo tqc --env SE-Forward-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 --save-freq 10000
```

For small slopes and symmetric control, it is recommended to specify a bigger `allowed_leg_angle` for the env: `allowed_leg_angle:25` (25deg instead of the 20deg by default)

Turn Left:
```
python train.py --algo tqc --env SE-TurnLeft-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:False --save-freq 10000
```

Turn Right without symmetry:
```
python train.py --algo tqc --env SE-TurnLeft-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:False task:"'turn_right'" --save-freq 10000
```

Full controller:
```
python train.py --algo tqc --env SE-MultiTask-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:False --save-freq 10000
```

Visualize a trained agent (400 steps):
```
python enjoy.py --algo tqc --env SE-Symmetric-v1 --num-threads 2 -f logs\ --load-last-checkpoint --exp-id 0 -n 400
```

## Slope and obstacles

The command for the sloped obstacles is the following:
```
/generate slopes c:9 e:0,0.1 g:2
```
- the c:9 argument says that you want 9 obstacles in total.. I recommend using only powers of 2

- the e:0,0.2 argument specifies the slopes
  - the first number is the slope in the right/left axis
  - the second number is the slope in the forward/backward

For example, the value 0,0.2 means that the obstacle isn't sloped in the right/left axis but in the forward/backward axis for every 1 meter forward, you do 0.2 meters up, so you end up with an uphill obstacle.
You can also use negative numbers and you can also combines the axes, for example, e:1,1 will go 45 degrees in both axes

- the last parameter, g:2, specifies the gravity.. g:1 is the default - 1g (9.81 m/(s*s))
but it might make sense to use e.g. 2g because with 1g the robot just slides on the terrain (even if it isn't sloped) and it looks quite unrealistic
(don't use more than g:5)


## Inverse kinematics (amp robot only)

To switch the IK dataset, go to the InverseKinematics folder, delete the current amp.txt file and replace it with either `amp_original.txt` or `amp_pointing_down.txt` (rename them to amp.txt because that's the name the game will be looking for)


## Gym Env Details

### Coordinate system

The coordinate system is changed from the one that is received:
- x axis is aligned with the right direction of the robot
- y axis is aligned with the forward direction of the robot
- z axis is pointing upward (opposite direction of gravity)

At every reset, the initial pose of the robot is used as reference for the following episode.
The current position of the robot in this new frame is named `self.world_position` in the code:

```python
self.world_position = np.zeros(3)  # x,y,z world position (centered at zero at reset)
self.robot_position = Point3D(np.zeros(3))  # x,y,z tracking position (without transform)
```
`self._update_world_position()` convert robot position to the new frame and compute delta with previous position.

### Observation Space

The observation space details can be found in `_extract_observation`, it mainly contains end-effector positions, velocities, robot rotation, heading deviation, angular velocity and input command (user command).

### Action Space

The action space consists of absolute end-effector desired positions and speed to reach the desired
position (4d per leg).
By default, the speed is limited to 10 (out of 100) but this can be changed by passing a higher value to `max_speed`.

The initial pose of the robot is slightly changed to improve stability: front legs are moved a bit forward and hind legs are moved a bit backward (looking more like a spider).

The action space is also restricted based on the initial pose (assumed stable) of the robot.
