import json
import math
import os
import random
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Tuple

import gym
import numpy as np
import zmq
from gym import spaces
from scipy.spatial.transform import Rotation as R

from gym_space_engineers.util.util import Point3D, in_relative_frame, normalize_angle


class Task(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class SymmetryType(Enum):
    LEFT_RIGHT = "left_right"
    PER_LEG = "per_leg"


class WalkingRobotIKEnv(gym.Env):
    """
    Gym interface to learn to walk.

    :param detach: for debug, it prevents the robot from moving away from its spawn position
    :param threshold_center_deviation: how far the robot may deviate from the center until the episode is stopped
    :param weight_center_deviation: weight for the off center derivation in y axis
    :param weight_distance_traveled: weight for the distance travelled in x axis
    :param weight_heading_deviation: weight for not walking with the right heading
    :param control_frequency: limit control frequency (in Hz)
    :param max_action: limit the legs to move ``max_action`` meters in each direction
    :param max_speed: limit the max speed of the legs
    :param desired_linear_speed: desired forward/backward speed in m/s
    :param desired_angular_speed: desired angular (left/right) speed in deg/s
    :param task: current task id, one of "forward", "backward", "turn_left", "turn_right"
    :param initial_wait_period: Time to wait for the initial reset in second
    :param symmetric_control: Reduces the search space by using symmetries
        (dependent on the task)
    :param allowed_leg_angle: Angle allowed around the starting position,
        this limits the action space
    :param symmetry_type: Type of symmetry to use.
        - "left_right": mirror right legs movements according to left leg movements
        - "per_leg": "triangle" symmetry, only control two legs
            and then mirror or copy for the rest
    :param verbose: control verbosity of the output (useful for debug)
    :param randomize_task: Whether to randomize the task being solved.
        For now, only randomize forward/backward or turn left/right,
        not all four at the same time.
    :param add_end_effector_velocity: Add end effector velocity to observation
    """

    def __init__(
        self,
        detach: bool = False,
        threshold_center_deviation: float = 10000,  # TODO(toni): tune it
        weight_center_deviation: float = 1,
        weight_distance_traveled: float = 5,
        weight_heading_deviation: float = 1,
        weight_turning_angle: float = 5,
        weight_linear_speed: float = 0.0,
        weight_angular_speed: float = 0.0,
        control_frequency: float = 10.0,
        max_action: float = 5.0,
        max_speed: float = 10.0,
        limit_control_freq: bool = True,
        desired_linear_speed: float = 3.0,  # in m/s (slow: 1 m/s, fast: 3-4 m/s)
        desired_angular_speed: float = 30.0,  # in deg/s (slow 5 deg/s, fast: 25-30 deg/s)
        task: str = "forward",
        initial_wait_period: float = 1.0,
        symmetric_control: bool = False,
        allowed_leg_angle: float = 15.0,  # in deg
        symmetry_type: str = "left_right",
        verbose: int = 1,
        randomize_task: bool = False,
        add_end_effector_velocity: bool = False,
    ):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        # Connect to server
        SERVER_ADDR = os.environ.get("SE_SERVER_ADDR", "localhost:5560")
        self.socket.connect(f"tcp://{SERVER_ADDR}")

        self.detach = detach
        self.id = None  # client id

        # Target control frequency in Hz
        self.limit_control_freq = limit_control_freq
        self.control_frequency = control_frequency
        self.wanted_dt = 1.0 / self.control_frequency
        self.current_sleep_time = self.wanted_dt
        # To avoid unbounded error:
        self.max_dt = 2.0  # in s
        self.last_time = time.time()
        self._first_step = True
        self.dt = 0.0

        self.initial_wait_period = initial_wait_period
        self.symmetric_control = symmetric_control
        self.symmetry_type = SymmetryType(symmetry_type)

        # TODO: contact indicator / torque ?

        # For now, this is hardcoded for the 6-legged robot
        self.number_of_legs = 6
        self.num_dim_per_leg = 4

        self.desired_linear_speed = desired_linear_speed
        self.desired_angular_speed = np.deg2rad(desired_angular_speed)
        # Desired delta in angle (in rad)
        self.desired_angle_delta = self.desired_angular_speed * self.wanted_dt
        self.add_end_effector_velocity = add_end_effector_velocity

        try:
            self.task = Task(task)
        except ValueError:
            raise ValueError(f"`task` must be one of {list(Task)}, not {task}")

        self.randomize_task = randomize_task
        # Tasks to randomize
        if self.task in [Task.FORWARD, Task.BACKWARD]:
            self.tasks = [Task.FORWARD, Task.BACKWARD]
        else:
            self.tasks = [Task.TURN_LEFT, Task.TURN_RIGHT]

        # Observation space dim
        num_var_per_joint = 0  # position,velocity,torque?
        dim_joints = self.number_of_legs * self.num_dim_per_leg * num_var_per_joint
        dim_velocity = 3 + 3  # Linear and angular velocity
        dim_current_rotation = 3
        dim_heading = 1  # deviation to desired heading
        dim_end_effector = self.number_of_legs * 3
        dim_command = 2  # forward/backward + left/right
        dim_additional = dim_heading + dim_end_effector + dim_command

        if add_end_effector_velocity:
            dim_additional += dim_end_effector

        self.input_dimension = dim_joints + dim_velocity + dim_current_rotation + dim_additional

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dimension,))

        # For now, we expect that the legs can move 5 meters in each direction
        # We use this value to map the [-1,1] interval to the actual reachable space
        self.max_action = max_action
        self.min_speed = 0
        self.max_speed = max_speed
        self.action_dim = self.number_of_legs * self.num_dim_per_leg

        self.action_upper_limits = np.ones(self.number_of_legs * self.num_dim_per_leg)
        self.action_lower_limits = np.ones(self.number_of_legs * self.num_dim_per_leg)

        # The end effector position is defined with respect to the shoulder
        # x: aligned with the "right" direction of the robot
        # y: pointing downward (aligned with gravity)
        # z: aligned with the "forward" direction of the robot
        # Note: z is with respect to the center of the mech for now

        # Get leg length by sending initial request
        response = self._send_initial_request()
        # Initialize variables
        self.last_end_effector_pos = np.stack([self.to_array(pos) for pos in response["endEffectorPositions"]])
        # Approximate leg length
        y_init = leg_length = abs(response["endEffectorPositions"][0]["y"])
        allowed_angle = np.deg2rad(allowed_leg_angle)
        # Compute allowed delta in action space w.r.t. the scale of the robot
        delta_allowed = leg_length * np.tan(allowed_angle)

        # We assume symmetric shape (similar legs)
        x_init = abs(response["endEffectorPositions"][0]["x"])

        # Limit Y axis to be at most y_max
        self.action_upper_limits[1 :: self.num_dim_per_leg] = -y_init / 2
        # Limit Y axis to be at least above initial pos
        self.action_lower_limits[1 :: self.num_dim_per_leg] = -y_init

        # Limit Left legs x axis
        self.action_lower_limits[0 : self.action_dim // 2 : self.num_dim_per_leg] = -x_init - delta_allowed
        self.action_upper_limits[0 : self.action_dim // 2 : self.num_dim_per_leg] = min(-x_init + delta_allowed, 0.0)
        # Limit Right legs x axis
        self.action_lower_limits[self.action_dim // 2 :: self.num_dim_per_leg] = max(0.0, x_init - delta_allowed)
        self.action_upper_limits[self.action_dim // 2 :: self.num_dim_per_leg] = x_init + delta_allowed

        # NOTE: it seems that z init is different for each leg
        z_inits = np.array([response["endEffectorPositions"][i]["z"] for i in range(self.number_of_legs)])
        # Offset default z to have a more stable starting pose
        z_offsets = 2 * np.array([-1, 0, 1, -1, 0, 1])
        z_inits += z_offsets
        # Limit z axis movement for all legs
        self.action_lower_limits[2 :: self.num_dim_per_leg] = z_inits - delta_allowed
        self.action_upper_limits[2 :: self.num_dim_per_leg] = z_inits + delta_allowed

        # Update limits for speed input
        self.action_upper_limits[self.num_dim_per_leg - 1 :: self.num_dim_per_leg] = self.max_speed
        self.action_lower_limits[self.num_dim_per_leg - 1 :: self.num_dim_per_leg] = self.min_speed

        # [X, Y, Z, Speed] for each of the 6 legs
        # (X, Y, Z) is a position relative to the shoulder joint of each leg
        # This position will be given to the inverse kinematics model
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.action_upper_limits.shape,
            dtype=np.float32,
        )

        if self.symmetric_control:
            if self.symmetry_type == SymmetryType.LEFT_RIGHT:
                # Half the size
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(len(self.action_upper_limits) // 2,),
                    dtype=np.float32,
                )
            else:
                # Control only two legs
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2 * self.num_dim_per_leg,),
                    dtype=np.float32,
                )

        # Weights for the different reward terms
        # self.weight_continuity = weight_continuity
        self.weight_center_deviation = weight_center_deviation
        self.weight_distance_traveled = weight_distance_traveled
        self.weight_heading_deviation = weight_heading_deviation
        self.weight_turning_angle = weight_turning_angle
        self.weight_linear_speed = weight_linear_speed
        self.weight_angular_speed = weight_angular_speed
        self.threshold_center_deviation = threshold_center_deviation

        # Early termination condition and costs
        self.early_termination_penalty = 100  # 1000 when using desired speed
        # Allow the robot to deviate 45deg from initial orientation before
        # terminating an episode
        self.heading_deviation_threshold_radians = np.deg2rad(45.0)

        # Limit to consider the robot has fallen
        # Lower this value to have a more stable walking gait
        self.roll_over_limit = np.deg2rad(40)
        # Height limit to assume that the robot is crawling
        # self.crawling_height_limit = 0.08

        # Control verbosity of the output
        self.verbose = verbose

        # holds all the necessary information
        self.heading = 0  # heading in radians
        self.start_heading = 0.0
        self.target_heading = 0.0  # when turning
        self.last_heading = 0.0
        self.current_rot = np.zeros(3)
        self.last_rot = np.zeros(3)

        self.world_position = np.zeros(3)  # x,y,z world position (centered at zero at reset)
        self.robot_position = Point3D(np.zeros(3))  # x,y,z tracking position (without transform)
        self.old_world_position = Point3D(np.zeros(3))
        self.delta_world_position = Point3D(np.zeros(3))  # x,y,z world position change from last position
        self.rotation_matrix = np.eye(3)
        self.translation = Point3D(np.zeros(3))
        # Angular velocity
        self.ang_vel = np.zeros(3)
        self._last_response = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.id is None:
            raise Exception("Please call reset() before step()")

        if self.symmetric_control:
            # Extend to match the required action dim
            if self.symmetry_type == SymmetryType.PER_LEG:
                n_repeat = (self.number_of_legs * self.num_dim_per_leg) // len(action)
                action = np.tile(action, n_repeat)
                # FIXME: remove that when z is the same for all legs
                action = self.apply_symmetry(action)
            else:
                action = np.array([action, action]).flatten()

        # The agent outputs a scaled action in [-1, 1]
        scaled_action = action.copy()
        # Unscale to real action
        action = self.unscale_action(action)

        if self.symmetric_control and self.symmetry_type == SymmetryType.LEFT_RIGHT:
            action = self.apply_symmetry(action)

        commands = {}
        leg_ids = ["l1", "l2", "l3", "r1", "r2", "r3"]

        for i, leg_id in enumerate(leg_ids):
            # Extract action values for each leg
            start_idx = self.num_dim_per_leg * i
            values = action[start_idx : start_idx + self.num_dim_per_leg]
            commands[leg_id] = {
                "position": {
                    "x": values[0],
                    "y": values[1],
                    "z": values[2],
                },
                # allow to cap the speed externally to keep current pose
                "speed": min(values[3], self.max_speed),
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
        reward = self._compute_reward(scaled_action, done)

        info = {
            # "up": up,
            # "forward": forward,
        }

        info.update(self._additional_infos())

        return observation, reward, done, info

    def apply_symmetry(self, action: np.ndarray) -> np.ndarray:
        right_start_idx = self.action_dim // 2

        if self.symmetry_type == SymmetryType.LEFT_RIGHT:
            # Note: the symmetric control on scaled actions does not seem to work as well
            # (bias towards going backward)
            action[right_start_idx :: self.num_dim_per_leg] = -action[0 : right_start_idx : self.num_dim_per_leg]
            # Same y and speed
            action[right_start_idx + 1 :: self.num_dim_per_leg] = action[1 : right_start_idx : self.num_dim_per_leg]
            action[right_start_idx + 3 :: self.num_dim_per_leg] = action[3 : right_start_idx : self.num_dim_per_leg]
            if self.task in [Task.FORWARD, Task.BACKWARD]:
                # Same z
                action[right_start_idx + 2 :: self.num_dim_per_leg] = action[2 : right_start_idx : self.num_dim_per_leg]
            elif self.task in [Task.TURN_LEFT, Task.TURN_RIGHT]:
                # Opposite z
                # Note: symmetric action on scaled action seems to work better for turning
                action[right_start_idx + 2 :: self.num_dim_per_leg] = -action[2 : right_start_idx : self.num_dim_per_leg]
        else:
            first_leg = action[: self.num_dim_per_leg]
            second_leg = action[self.num_dim_per_leg : 2 * self.num_dim_per_leg]
            second_leg[0] = -1
            # Indices for each leg
            start_indices = np.arange(self.number_of_legs * self.num_dim_per_leg, step=self.num_dim_per_leg)
            # Copy for the same side
            start_idx_1 = start_indices[::2]
            for i in range(self.num_dim_per_leg):
                action[start_idx_1 + i] = first_leg[i]

            # Opposite x for opposite side
            indices = start_idx_1[start_idx_1 >= right_start_idx]
            action[indices] = -action[indices]

            start_idx_2 = start_indices[1::2]
            for i in range(self.num_dim_per_leg):
                action[start_idx_2 + i] = second_leg[i]

            # Opposite x for opposite side
            indices = start_idx_2[start_idx_2 < right_start_idx]
            action[indices] = -action[indices]

        return action

    def reset(self) -> np.ndarray:
        # Reset values for controlling frequency
        self._first_step = True
        self.current_sleep_time = self.wanted_dt
        self.last_time = time.time()

        # Select a task randomly
        if self.randomize_task:
            self.task = random.choice(self.tasks)

        if self.id is None:
            response = self._send_initial_request()
        else:
            direction = "backward" if self.task == Task.BACKWARD else "forward"
            request = {
                "id": self.id,
                "type": "Reset",
                "blueprintDirection": direction,
            }
            response = self._send_request(request)

        # IMPORTANT: update robot pose before reseting the transform
        self._update_robot_pose(response)
        self.old_world_position = Point3D(np.zeros(3))
        self._reset_transform()

        return self._get_observation(response)

    def change_task(self, task: Task) -> np.ndarray:
        # The reset transform would break without info about the robot
        assert self._last_response is not None
        assert isinstance(task, Task)
        self.task = task
        self._update_robot_pose(self._last_response)
        self.old_world_position = Point3D(np.zeros(3))
        self._reset_transform()
        return self._get_observation(self._last_response)

    def _update_robot_pose(self, response: Dict[str, Any]) -> None:
        position = self.to_array(response["position"])
        right = self.to_array(response["right"])
        forward = self.to_array(response["forward"])
        up = self.to_array(response["up"])

        rot_mat = R.from_matrix(np.array([right, forward, up]).T)
        self.current_rotation_matrix = rot_mat.as_matrix()
        self.last_rot = self.current_rot.copy()
        self.current_rot = rot_mat.as_euler("xyz", degrees=False)
        self.heading = normalize_angle(self.current_rot[2])  # extract yaw
        self.last_heading = normalize_angle(self.last_rot[2])
        # self.ang_vel = np.array(response["ang_vel"])
        self.robot_position = Point3D(position)

    def _get_observation(self, response: Dict[str, Any]) -> np.ndarray:
        # Extract response from server
        self._update_robot_pose(response)
        self._update_world_position()
        self.dt = self._update_control_frequency()
        # Update target heading
        desired_delta = self.desired_angle_delta
        if self.task == Task.TURN_RIGHT:
            desired_delta *= -1

        self.target_heading = normalize_angle(self.heading + desired_delta)

        observation = self._extract_observation(response)

        return observation

    def _extract_observation(self, response: Dict[str, Any]) -> np.ndarray:
        # lin_acc = np.array(response["lin_acc"])
        # joint_torque = np.array(response["joint_torque"])
        # joint_positions = np.array(response["joint_positions"])
        # joint_velocities = np.array(response["joint_velocities"])
        end_effector_positions = np.stack([self.to_array(pos) for pos in response["endEffectorPositions"]])
        # Use finite difference
        velocity = np.array(self.delta_world_position) / self.wanted_dt
        angular_velocity = (self.current_rot - self.last_rot) / self.wanted_dt
        end_effector_velocity = (end_effector_positions - self.last_end_effector_pos) / self.wanted_dt
        self.last_end_effector_pos = end_effector_positions.copy()

        if self.task in [Task.FORWARD, Task.BACKWARD]:
            # TODO: clip target heading to max heading deviation when using the model?
            heading_deviation = normalize_angle(self.heading - self.start_heading)
        elif self.task in [Task.TURN_LEFT, Task.TURN_RIGHT]:
            # Note: this is only needed in the case of precise turning
            heading_deviation = normalize_angle(self.heading - self.target_heading)

        # Append input command, one for forward/backward
        # one for turn left/right
        # TODO(toni): allow a mix of commands
        input_command = {
            Task.FORWARD: [1, 0],
            Task.BACKWARD: [-1, 0],
            Task.TURN_LEFT: [0, 1],
            Task.TURN_RIGHT: [0, -1],
        }[self.task]

        if self.add_end_effector_velocity:
            end_effector_velocity = end_effector_velocity.flatten()
        else:
            end_effector_velocity = np.array([])

        observation = np.concatenate(
            (
                # TODO(toni): check normalization
                # TODO: check z definition (absolute or relative)
                end_effector_positions.flatten() / self.max_action,
                end_effector_velocity,
                self.current_rot,
                velocity,
                angular_velocity,
                # TODO(toni): add center deviation?
                # joint_torque,
                # joint_positions,
                # joint_velocities,
                # lin_acc,
                np.array([heading_deviation]),
                # np.array([heading_deviation, self.dt]),
                np.array(input_command),
            )
        )
        return observation

    def _update_control_frequency(self, reset: bool = False) -> float:
        # Limit controller frequency
        # Update control frequency estimate
        # clip to account for crashes
        dt = np.clip(time.time() - self.last_time, 0.0, self.max_dt)
        self.last_time = time.time()

        if not self._first_step:
            # compute error in control frequency
            # positive: the control loop is too fast: need to sleep a bit
            # negative: the control loop is too slow: do not sleep
            control_dt_error = self.wanted_dt - dt
            # clip the error
            control_dt_error = np.clip(control_dt_error, -0.5 * self.wanted_dt, 0.5 * self.wanted_dt)

            corrected_sleep_time = self.current_sleep_time + control_dt_error

            # gradually_update
            alpha_sleep = 0.1
            self.current_sleep_time = corrected_sleep_time * alpha_sleep + (1 - alpha_sleep) * self.current_sleep_time

            # Clip again
            self.current_sleep_time = np.clip(self.current_sleep_time, 0.0, self.wanted_dt)
        else:
            # First step: the dt would be zero
            self._first_step = False

        if self.verbose > 1:
            print(f"{1 / dt:.2f}Hz")
        if self.limit_control_freq:
            time.sleep(self.current_sleep_time)
        return dt

    @staticmethod
    def to_array(vector: Dict[str, np.ndarray]) -> np.ndarray:
        # return np.array([vector["x"], vector["y"], vector["z"]])
        # Re-arrange to match convention
        return np.array([vector["z"], vector["x"], vector["y"]])

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_message = json.dumps(request)
        self.socket.send(request_message.encode("UTF-8"))
        response = json.loads(self.socket.recv())
        # Cache last response, useful when changing tasks on the fly
        self._last_response = deepcopy(response)
        return response

    def _send_initial_request(self) -> Dict[str, Any]:
        direction = "backward" if self.task == Task.BACKWARD else "forward"
        request = {
            "type": "Initial",
            "blueprintName": "Mech-v0-NS-AM",
            "environment": "Obstacles3",
            "initialWaitPeriod": self.initial_wait_period,
            "detach": self.detach,
            "blueprintDirection": direction,
        }
        response = self._send_request(request)
        self.id = response["id"]

        return response

    def render(self, mode="human"):
        pass

    def close(self):
        if self.id is not None:
            request = {
                "type": "Stop",
                "id": self.id,
            }
            try:
                self._send_request(request)
                self.socket.close()
            except zmq.error.ZMQError:
                pass

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
        # self.translation.z = 0  # don't move in z
        self.start_heading = self.heading
        self.rotation_matrix = self.current_rotation_matrix.copy()
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

        if self.task in [Task.FORWARD, Task.BACKWARD]:
            reward = self._compute_walking_reward(scaled_action, done)
        elif self.task in [Task.TURN_LEFT, Task.TURN_RIGHT]:
            reward = self._compute_turning_reward(scaled_action, done)
        return reward

    def _compute_turning_reward(self, scaled_action: np.ndarray, done: bool) -> float:
        deviation_cost = self.weight_center_deviation * self._xy_deviation_cost()
        # angular_velocity_cost = self.weight_angular_velocity * self._masked_angular_velocity_cost()
        # continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)
        continuity_cost = 0.0

        # use delta in orientation as primary reward
        # the sign of the desired delta make the robot rotate clockwise or anti-clockwise
        delta_heading_rad = normalize_angle(self.heading - self.last_heading)
        delta_heading = np.rad2deg(delta_heading_rad)

        desired_delta = self.desired_angle_delta
        if self.task == Task.TURN_RIGHT:
            desired_delta *= -1

        # For debug, to calibrate target speed
        if self.verbose > 1:
            current_speed = delta_heading / self.wanted_dt
            print(f"Angular Speed: {current_speed:.2f} deg/s")

        angular_speed_cost = (delta_heading_rad - desired_delta) ** 2 / self.desired_angle_delta ** 2
        angular_speed_cost = self.weight_angular_speed * angular_speed_cost

        # Clip to be at most desired_delta
        if self.weight_angular_speed > 0:
            desired_delta_deg = np.rad2deg(desired_delta)
            delta_heading = np.clip(delta_heading, -desired_delta_deg, desired_delta_deg)

        turning_reward = delta_heading * self.weight_turning_angle
        if self.task == Task.TURN_RIGHT:
            turning_reward *= -1

        # if self.verbose > 1:
        #     print(f"Turning Reward: {turning_reward:.5f}", f"Continuity Cost: {continuity_cost:5f}")
        #     print(f"Deviation cost: {deviation_cost:.5f}")
        #     print(f"Angular velocity cost: {angular_velocity_cost:.5f}")

        # Do not reward agent if it has terminated due to fall/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            turning_reward = 0.0

        reward = -(deviation_cost + angular_speed_cost + continuity_cost) + turning_reward

        if done:
            # give negative reward
            reward -= self.early_termination_penalty
        return reward

    def _xy_deviation_cost(self) -> float:
        """
        Cost for deviating from the center of the treadmill (y = 0)
        :return: normalized squared value for deviation from a straight line
        """
        # TODO: tune threshold_center_deviation
        # Note: it is used a bit differently for walking/turning
        # maybe better to have two variables
        deviation = self._rotation_center_deviation() / self.threshold_center_deviation
        return deviation ** 2

    def _rotation_center_deviation(self) -> float:
        return np.sqrt(self.world_position.x ** 2 + self.world_position.y ** 2)

    def _compute_walking_reward(self, scaled_action: np.ndarray, done: bool) -> float:
        deviation_cost = self.weight_center_deviation * self._center_deviation_cost()

        # continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)
        # Note: we are not using continuity cost for now, as energy efficiency is not needed in simulation
        continuity_cost = 0.0
        heading_cost, is_headed = self._heading_cost()
        heading_cost *= self.weight_heading_deviation

        # Desired delta in distance
        desired_delta = self.desired_linear_speed * self.wanted_dt

        if self.task == Task.BACKWARD:
            desired_delta *= -1

        # For debug, to calibrate target speed
        if self.verbose > 1:
            current_speed = self.delta_world_position.y / self.wanted_dt
            print(f"Speed: {current_speed:.2f} m/s")

        linear_speed_cost = (desired_delta - self.delta_world_position.y) ** 2 / desired_delta ** 2
        linear_speed_cost = self.weight_linear_speed * linear_speed_cost

        distance_traveled = self.delta_world_position.y
        # Clip to be at most desired_delta
        if self.weight_linear_speed > 0.0:
            distance_traveled = np.clip(distance_traveled, -desired_delta, desired_delta)

        # use delta in y direction as distance that was travelled
        distance_traveled_reward = distance_traveled * self.weight_distance_traveled

        if self.task == Task.BACKWARD:
            distance_traveled_reward *= -1

        # Do not reward agent if it has terminated due to fall/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            distance_traveled_reward = 0.0

        if self.verbose > 1:
            # f"Continuity Cost: {continuity_cost:5f}
            print(f"Linear Speed Cost: {linear_speed_cost:.5f}")
            print(f"Deviation cost: {deviation_cost}")
            print(f"Heading cost: {heading_cost}")

        reward = distance_traveled_reward + -(deviation_cost + heading_cost + continuity_cost + linear_speed_cost)
        if done:
            # give negative reward
            reward -= self.early_termination_penalty
        return reward

    def _center_deviation_cost(self) -> float:
        """
        Cost for deviating from the center of the track (y = 0)

        :return: normalized squared value for deviation from a straight line
        """
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
        return bool(
            math.fabs(self.current_rot[0]) > self.roll_over_limit or math.fabs(self.current_rot[1]) > self.roll_over_limit
        )

    def is_crawling(self) -> bool:
        """
        :return True if the robot is too low
        """
        # NOTE: probably world_position is fine here
        return bool(self.robot_position.z < self.crawling_height_limit)

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        # Deactivate crawling detection for sim
        # is_crawling = self.is_crawling()
        is_crawling = False
        has_fallen = self.has_fallen()
        if self.task in [Task.FORWARD, Task.BACKWARD]:
            is_centered = math.fabs(self.world_position.x) < self.threshold_center_deviation
            _, is_headed = self._heading_cost()
        elif self.task in [Task.TURN_LEFT, Task.TURN_RIGHT]:
            is_centered = self._rotation_center_deviation() < self.threshold_center_deviation
            is_headed = True

        return has_fallen or not is_centered or not is_headed or is_crawling


if __name__ == "__main__":
    import gym

    # noinspection PyUnresolvedReferences
    import gym_space_engineers  # noqa: F401

    def postprocess_action(action):
        # Multiply x by -1 for the legs on the right side
        action[3:, 0] *= -1

        # Divide x,y,z by 10 to fit them into [-1,1]
        action[:, 0:3] /= 10

    for _ in range(1):
        env = gym.make("SpaceEngineers-WalkingRobot-IK-v0", detach=False)

        observation = env.reset()
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

        observation = env.reset()
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
