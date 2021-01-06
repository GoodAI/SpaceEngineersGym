import gym
import numpy as np
from gym import spaces
from gym_se.agent import AgentController, MoveArgs


class SpaceEngineersEnv(gym.Env):
    """
    Simple Space Engineers environment where the agent moves in a maze.

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
    """


    def __init__(self):
        self.agent = AgentController()

        # Base position is the center
        self.base_position = np.array([-510.71, 379, 385.20])
        self.position = np.array([0, 0])

        self.observation_space = spaces.Box(low=np.zeros(16), high=np.ones(16), dtype=np.float32)

        self.action_bounds = 1
        self.action_space = spaces.Box(low=np.array([-self.action_bounds, -self.action_bounds]),
                                       high=np.array([self.action_bounds, self.action_bounds]), dtype=np.float32)

    @staticmethod
    def _get_position(observation):
        position_raw = observation["Position"]
        position = np.array([position_raw["X"], position_raw["Y"], position_raw["Z"]])
        return position

    @staticmethod
    def _get_2d_position(observation):
        position_raw = observation["Position"]
        position = np.array([position_raw["X"], position_raw["Z"]])
        return position

    @staticmethod
    def _get_move_args(vector):
        return MoveArgs(float(vector[0]), float(vector[1]), float(vector[2]))

    @staticmethod
    def _vector2_to_vector3(vector):
        return MoveArgs(float(vector[0]), 0, float(vector[1]))

    def _get_observation(self, observation):
        position = self._get_position(observation) - self.base_position
        position_2d = np.array([position[0], position[2]])
        raycast_results = np.array(observation["RayCastResults"])
        concat = np.concatenate((position_2d, raycast_results))

        return concat

    def step(self, action):
        action = np.clip(action, -self.action_bounds, self.action_bounds)
        observation = self.agent.move(self._vector2_to_vector3(action))
        observation = self._get_observation(observation)
        reward = abs(observation[0]) + abs(observation[1])

        # return observation, reward, isDone, info
        return observation[2:], reward, False, {
            'position': observation[0:2]
        }

    def reset(self):
        observation = self.agent.teleport(self._get_move_args(self.base_position))
        observation = self._get_observation(observation)

        return observation[2:]

    def render(self, mode='human'):
        ...

    def close(self):
        self.agent.close_connection()


# Test the environment by doing 1000 random steps in the game
if __name__ == "__main__":
    import gym

    env = gym.make('SpaceEngineers-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        observation, _, _, _ = env.step(env.action_space.sample())
        print(observation)
    env.close()
