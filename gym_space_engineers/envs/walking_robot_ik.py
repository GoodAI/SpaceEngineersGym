import json
import math
from typing import Any, Dict, Tuple

import gym
import numpy as np
import zmq
from gym import spaces
from scipy.spatial.transform import Rotation as R

from gym_space_engineers.util.util import Point3D, in_relative_frame, normalize_angle

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5560")


class WalkingRobotIKEnv(gym.Env):
    """
    Gym interface to learn to walk.

    :param detach: for debug, it prevents the robot from moving away from its spawn position
    :param threshold_center_deviation: how far the robot may deviate from the center until the episode is stopped
    :param weight_center_deviation: weight for the off center derivation in y axis
    :param weight_distance_traveled: weight for the distance travelled in x axis
    :param weight_heading_deviation: weight for not walking with the right heading
    :param weight_angular_velocity: weight for any angular velocity
    :param verbose: control verbosity of the output (useful for debug)
    """

    def __init__(
        self,
        detach: bool = True,
        threshold_center_deviation: float = 10000,  # TODO(toni): tune it
        weight_center_deviation: float = 1,
        weight_distance_traveled: float = 50,
        weight_heading_deviation: float = 1,
        weight_angular_velocity: float = 1.0,
        verbose: int = 1,
    ):
        self.detach = detach
        self.id = None
        self.observation_space = None

        # For now, this is hardcoded for the 6-legged robot
        self.number_of_legs = 6
        self.num_dim_per_leg = 4
        self.max_action = 1
        # TODO(toni): update lower and upper limits
        self.action_upper_limits = np.ones(self.number_of_legs * self.num_dim_per_leg) * self.max_action
        self.action_lower_limits = np.ones(self.number_of_legs * self.num_dim_per_leg) * -self.max_action

        # For now, we expect that the legs can move 10 meters in each direction
        # We use this value to map the [-1,1] interval to the actual reachable space
        self.action_space_size = 10

        # [X, Y, Z, Speed] for each of the 6 legs
        # (X, Y, Z) is a position relative to the shoulder joint of each leg
        # This position will be given to the inverse kinematics model
        self.action_space = spaces.Box(
            low=np.stack([[-1, -1, -1, -1] for _ in range(self.number_of_legs)]).flatten(),
            high=np.stack([[1, 1, 1, 1] for _ in range(self.number_of_legs)]).flatten(),
            dtype=np.float32,
        )

        # Weights for the different reward terms
        # self.weight_continuity = weight_continuity
        self.weight_center_deviation = weight_center_deviation
        self.weight_distance_traveled = weight_distance_traveled
        self.weight_heading_deviation = weight_heading_deviation
        self.weight_angular_velocity = weight_angular_velocity
        self.threshold_center_deviation = threshold_center_deviation

        # Early termination condition and costs
        self.early_termination_penalty = 2
        # Allow the robot to deviate 45deg from initial orientation before
        # terminating an episode
        self.heading_deviation_threshold_radians = np.deg2rad(45.0)

        # Limit to consider the robot has fallen
        # Lower this value to have a more stable walking gait
        self.roll_over_limit = np.deg2rad(40)
        # Height limit to assume that the robot is crawling
        # self.crawling_height_limit = 0.08

        self.verbose = verbose

        # holds all the necessary information
        self.heading = 0  # heading in radians
        self.start_heading = 0
        self.current_rot = np.zeros(3)
        # self.imu_orientation = np.zeros(3)
        # self.start_imu_orientation = np.zeros(3)
        self.world_position = np.zeros(3)  # x,y,z world position
        self.robot_position = Point3D(np.zeros(3))  # x,y,z tracking position (without transform)
        self.old_world_position = Point3D(np.zeros(3))
        self.delta_world_position = Point3D(np.zeros(3))  # x,y,z world position change from last position
        self.rotation_matrix = np.eye(3)
        self.translation = Point3D(np.zeros(3))
        # Angular velocity
        self.ang_vel = np.zeros(3)

    def step(self, action):
        if self.id is None:
            raise Exception("Please call reset() before step()")

        scaled_action = action
        # Unscale to real action
        action = self.unscale_action(action)

        commands = {}
        leg_ids = ["l1", "l2", "l3", "r1", "r2", "r3"]

        for i, leg_id in zip(range(self.number_of_legs), leg_ids):
            values = [(action[4 * i + j].item()) for j in range(4)]
            commands[leg_id] = {
                "position": {
                    "x": values[0] * self.action_space_size,
                    "y": values[1] * self.action_space_size,
                    "z": values[2] * self.action_space_size,
                },
                "speed": (values[3] + 1) * 50,
            }

        request = {
            "id": self.id,
            "type": "Command",
            "commands": commands,
        }

        response = self._send_request(request)
        observation = self._get_observation(response)

        # Update internal state if needed
        # (for instance n steps at targets, that should be decoupled from compute reward)
        self._on_step()
        done = self.is_terminal_state()
        reward = self.compute_reward(scaled_action, done)

        info = {
            # "up": up,
            # "forward": forward,
        }

        info.update(self._additional_infos())

        return observation, reward, done, info

    def reset(self):
        if self.id is None:
            response = self._send_initial_request()
        else:
            request = {
                "id": self.id,
                "type": "Reset",
            }
            response = self._send_request(request)
        # TODO: return initial observation
        self.old_world_position = Point3D(np.zeros(3))
        self._reset_transform()
        return self._get_observation(response)


    def _get_observation(self, response):
        # Extract response from server
        position = self._get_np_array_from_vector(response["position"])
        right = self._get_np_array_from_vector(response["right"])
        forward = self._get_np_array_from_vector(response["forward"])
        up = self._get_np_array_from_vector(response["up"])
        end_effector_positions = np.stack(self._get_array_from_vector(pos) for pos in response["endEffectorPositions"])

        # TODO(toni): find right convention
        rot_mat = R.from_matrix([right, forward, up])
        self.current_rot = rot_mat.as_euler("xyz", degrees=False)
        self.heading = normalize_angle(self.current_rot[2])  # extract yaw
        # self.ang_vel = np.array(response["ang_vel"])

        self.robot_position = Point3D(position)

        self._update_world_position()

        # dt = self.update_control_frequency(command)

        observation = self.extract_observation(response)

        # Save last observation in case no manual reset is needed
        self.last_obs = observation.copy()
        return observation

    def extract_observation(self, response):
        # lin_acc = np.array(response["lin_acc"])
        # joint_torque = np.array(response["joint_torque"])
        # joint_positions = np.array(response["joint_positions"])
        # joint_velocities = np.array(response["joint_velocities"])

        heading_deviation = normalize_angle(self.heading - self.start_heading)

        observation = np.concatenate(
            (
                self.current_rot,
                # joint_torque,
                # joint_positions,
                # joint_velocities,
                # self.ang_vel,
                # lin_acc,
                np.array([heading_deviation]),
                # np.array([heading_deviation, dt]),
            )
        )
        return observation

    @staticmethod
    def _get_array_from_vector(vector: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([vector["x"], vector["y"], vector["z"]])

    @staticmethod
    def _send_request(request):
        request_message = json.dumps(request)
        socket.send(request_message.encode("UTF-8"))
        response = json.loads(socket.recv())

        return response

    def _send_initial_request(self):
        request = {
            "type": "Initial",
            "detach": self.detach,
        }
        response = self._send_request(request)
        self.id = response["id"]

        return response

    def render(self, mode='human'):
        pass

    def close(self):
        if self.id is not None:
            request = {
                "type": "Stop",
                "id": self.id,
            }
            self._send_request(request)

    def _additional_infos(self) -> Dict[str, Any]:
        return {}

    def _on_step(self) -> None:
        pass

    def _update_world_position(self) -> None:
        self.world_position = Point3D(
            in_relative_frame(
                self.robot_position.toarray(),
                self.rotation_matrix,
                self.translation,
            )
        )
        self.delta_world_position = self.world_position - self.old_world_position
        self.old_world_position = self.world_position

    def _reset_transform(self) -> None:
        # use the starting position to initialize translation and rotation matrix
        self.translation = -self.robot_position
        # NOTE: We assume flat ground
        self.translation.z = 0  # don't move in z
        self.start_heading = self.heading
        self.rotation_matrix = R.from_euler("z", -self.heading, degrees=False).as_matrix()

        self._update_world_position()
        self.delta_world_position = Point3D(np.zeros(3))

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        """
        return 2.0 * ((action - self.action_lower_limits) / (self.action_upper_limits - self.action_lower_limits)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.action_lower_limits + (0.5 * (scaled_action + 1.0) * (self.action_upper_limits - self.action_lower_limits))

    def _compute_reward(self, scaled_action: np.ndarray, done: bool) -> float:

        deviation_cost = self.weight_center_deviation * self._center_deviation_cost()
        angular_velocity_cost = self.weight_angular_velocity * self._angular_velocity_cost()

        # continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)
        # Note: we are not using continuity cost for now, as energy efficiency is not needed in simulation
        continuity_cost = 0.0
        heading_cost, is_headed = self._heading_cost()
        heading_cost *= self.weight_heading_deviation
        # TODO(toni): check convention
        # use delta in x direction as distance that was travelled
        distance_reward = self.delta_world_position.z * self.weight_distance_traveled

        if self.verbose > 1:
            # f"Continuity Cost: {continuity_cost:5f}
            print(f"Distance Reward: {distance_reward:.5f}")
            print(f"Deviation cost: {deviation_cost}")
            print(f"Heading cost: {heading_cost}")

        # Do not reward agent if it has terminated due to fall/not headed/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            distance_reward = 0.0

        reward = -(deviation_cost + heading_cost + continuity_cost + angular_velocity_cost) + distance_reward
        if done:
            # give negative reward
            reward -= self.early_termination_penalty
        return reward

    def _center_deviation_cost(self) -> float:
        """
        Cost for deviating from the center of the track (y = 0)

        :return: normalized squared value for deviation from a straight line
        """
        # TODO(toni): adapt to correct direction
        deviation = self.world_position.x
        deviation = deviation / self.threshold_center_deviation
        return deviation ** 2

    def _heading_cost(self) -> Tuple[float, bool]:
        """
        Computes the deviation from the expected heading.

        :return: Normalized (0 to 1) squared deviation from expected heading and bool if it is still headed correctly
        """
        # assume heading and expected_heading is given in radians
        heading_offset = normalize_angle(self.heading - self.start_heading)
        heading_deviation = np.abs(heading_offset)
        heading_deviation = heading_deviation / self.heading_deviation_threshold_radians
        return heading_deviation ** 2, bool(heading_deviation < 1)

    def has_fallen(self) -> bool:
        """
        :return: True if the robot has fallen (roll or pitch above threshold)
        """
        # TODO(toni): check the indices
        return bool(
            math.fabs(self.current_rot[0]) > self.roll_over_limit or math.fabs(self.current_rot[1]) > self.roll_over_limit
        )

    def is_crawling(self) -> bool:
        """
        :return True if the robot is too low
        """
        # TODO(toni): check convention
        # NOTE: probably world_position is fine here
        return bool(self.robot_position.y < self.crawling_height_limit)

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        has_fallen = self.has_fallen()
        # TODO(toni): check convention
        is_centered = math.fabs(self.world_position.x) < self.threshold_center_deviation
        # Deactivate crawling detection for sim
        # is_crawling = self.is_crawling()
        is_crawling = False
        _, is_headed = self._heading_cost()
        return has_fallen or not is_centered or not is_headed or is_crawling


if __name__ == "__main__":
    import time

    import gym

    # noinspection PyUnresolvedReferences
    import gym_space_engineers  # noqa: F401

    def postprocess_action(action):
        # Multiply x by -1 for the legs on the right side
        action[3:, 0] *= -1

        # Divide x,y,z by 10 to fit them into [-1,1]
        action[:, 0:3] /= 10

    for _ in range(1):
        env = gym.make('SpaceEngineers-WalkingRobot-IK-v0', detach=False)

        observation, _, _, _ = env.reset()
        print(observation)

        # All legs from low to high
        for y in np.linspace(-10, 10, 30):
            env.render()

            left_leg_position = [-5, y, -0.5, 0.1]
            action = np.stack([left_leg_position for i in range(6)])
            postprocess_action(action)

            observation, _, _, _ = env.step(action.flatten())
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        observation, _, _, _ = env.reset()
        print(observation)

        # All legs from back to front
        for z in np.linspace(5, -10, 30):
            env.render()

            left_leg_position = [-5, -2, z, 0.1]
            action = np.stack([left_leg_position for i in range(6)])
            postprocess_action(action)

            observation, _, _, _ = env.step(action.flatten())
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        env.close()
