# Maze (ExploreEnvironments)

## How to load world

- Make sure that you are connected to Steam
- Run *VeriDream/StartMaze.bat*
- There will be a popup from Steam where you have to confirm that you want to run Space Engineers with some additional parameters
- The game should load the Maze world
- *(Optional)* Press enter and type */ToggleSensors* to show raycast sensors
- *(Optional)* Press enter and type */ToggleMaxSpeed* to uncap the speed of the game

## How to use the gym

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
The state consists of 16 rays that are cast from the player position to observe where are obstacles.
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