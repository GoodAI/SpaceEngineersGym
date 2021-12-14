# RL on top of manual controller

## Installation

- Clone and install `https://github.com/GoodAI/SpaceEngineersGym` (feat/manual-plus-rl branch)
- Clone and install `https://gitlab.goodai.com/veridream/rl-baselines3-zoo` (misc/veridream branch)

## How to run

For each of the commands below, you must be in the directory where the RL zoo is located.

- Train the *v0* robot with logging to tensorboard (single-task, forward)
    - `python train.py --algo tqc --env SE-Corrections-v1 --num-threads 2 --vec-env subproc -params n_envs:7 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:0 --save-freq 10000 --tensorboard-log logs/tensorboard/`
    - `n_envs` can be increased if the game can run at 60 UPS
    - `robot_id` can be changed to use a different robot
- Train the *v0* robot for turning
    - Use the `SE-Corrections-TurnLeft-v1` env instead
- Train the *v0* robot (multi-task)
    - `python train.py --algo tqc --env SE-Corrections-Multi-v1 --num-threads 2 --vec-env subproc -params n_envs:7 --eval-freq -1 --log-interval 8 --env-kwargs robot_id:0 --save-freq 10000 --tensorboard-log logs/tensorboard/`
- Visualize a trained agent
    - `python enjoy.py --algo tqc --env SE-Corrections-v1 --num-threads 2 -f logs\ --load-last-checkpoint --exp-id 0 -n 400`
    - change the `--exp-id`
    - 400 steps are approximately 2 episodes (can be increased to show more)
- Human control
    - Set the trained model as an env variable
        - e.g. `MULTI_CONTROLLER_PATH=logs/tqc/SE-Corrections-Multi-v1_5/rl_model_309876_steps.zip`
    - `python train.py --algo human --env SE-Corrections-Control-v1 --num-threads 2 --eval-freq -1 --log-interval 8 -params n_envs:1 --env-kwargs robot_id:0`
- Run tensorboard
    - `tensorboard --logdir logs/tensorboard/`
    - or just for a single env `tensorboard --logdir logs/tensorboard/SE-Corrections-v1`