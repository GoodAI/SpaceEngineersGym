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
python train.py --algo tqc --env SE-Symmetric-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:True --save-freq 10000
```

For small slopes and symmetric control, it is recommended to specify a bigger `allowed_leg_angle` for the env: `allowed_leg_angle:25` (25deg instead of the 15deg by default)

Turn Left:
```
python train.py --algo tqc --env SE-TurnLeft-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:True --save-freq 10000
```

Turn Right without symmetry:
```
python train.py --algo tqc --env SE-TurnLeft-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:False task:"'turn_right'" --save-freq 10000
```

Full controller:
```
python train.py --algo tqc --env SE-MultiTask-v1 --num-threads 2 --vec-env subproc -params n_envs:8 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:1 symmetric_control:True --save-freq 10000
```

Visualize a trained agent (400 steps):
```
python enjoy.py --algo tqc --env SE-Symmetric-v1 --num-threads 2 -f logs\ --load-last-checkpoint --exp-id 0 -n 400
```

## Slope and obstacles

The command for the sloped obstacles is the following:
```
/generate slopes c:9 e:0,0.2 g:2
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
