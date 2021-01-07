# SpaceEngineersGym

This repository contains an OpenAI Gym environment which makes it possible to control an agent in Space Engineers.

## Installation

- Download Space Engineers from [Steam](https://store.steampowered.com/app/244850/Space_Engineers/) and istall the game (primarily for Windows)
- Install [iv4xr-se-plugin](https://github.com/GoodAI/iv4xr-se-plugin) (follow the instructions in the plugin's repository)
- Install this module with `pip3 install .` (use -e flag if you want editable/link installation)
- Run the game at least once
- Download the [MazeWorld](https://drive.google.com/file/d/1He_0VkAvJpaqVyoYlTZZEn0TytRpU8mm/view?usp=sharing) archive
- Navigate to *%appdata%\SpaceEngineers\Saves\\<your steam id\>* and unzip the world there
- Run the game and load the MazeWorld scenario
- *(Optional)* Use the */ToggleMaxSpeed* command to increase the speed of the game

## How to use

The following snippet creates an instance of the gym environment and performs 1000 random steps in the game:

```python
import gym
import gym_space_engineers

env = gym.make('SpaceEngineers-v0')
env.reset()
for _ in range(1000):
    observation, _, _, _ = env.step(env.action_space.sample())
    print(observation)
env.close()
```

## Environment

Follows the description of the environment:

**STATE:**
The state consists of 16 rays that are cast from the player position the observe where are obstacles.
The rays are cast in uniform intervals - the difference between two consecutive rays is always 22.5 degrees.
The result of each ray cast is a number in the [0,1] interval.
    - Value of 0 means that there was an obstacle right in front of the agent.
    - Value of 1 means that there was either no obstacle or an obstacle at the maximum distance of 30 meters.

The position of the agent is not part of the state can be found in info["position"].
The position is a two-dimensional vector (x,y) where [0,0] is the position at the center of the maze.
The position is only two-dimensional because the agent currently cannot move in the third dimension.

**ACTIONS:**
The action is a two-dimension vector (x,y) where both x and y are in the [-1,1] range.
The x value controls how much to move along the x-axis.
The y value controls how much to move along the y-axis.

**NOTE:**
In order for this environment to work, the agent must be located in the maze world.
