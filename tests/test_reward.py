import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def plot_robot(vec_origin, vector, color="black"):
    plt.quiver(*vec_origin, *vector, color=color, scale=5)  # , units="xy"


plt.axis("equal")

heading_offset = np.deg2rad(90)

ninety_deg_rot = R.from_euler("z", np.deg2rad(90), degrees=False).as_matrix()[:2, :2]

init_heading = np.deg2rad(0.0)
current_rot = R.from_euler("z", init_heading, degrees=False)
current_rot_mat = current_rot.as_matrix()[:2, :2]

init_pos = np.array([-1, -1])
robot_x = current_rot_mat @ np.array([1, 0])
robot_y = ninety_deg_rot @ robot_x

plot_robot(init_pos, robot_x)
plot_robot(init_pos, robot_y)

desired_delta_heading = np.deg2rad(40)
desired_heading = init_heading + desired_delta_heading
desired_forward_distance = 4
desired_rot = R.from_euler("z", desired_heading, degrees=False)
desired_rot_mat = desired_rot.as_matrix()[:2, :2]

desired_new_x = desired_rot_mat @ np.array([1, 0])
desired_new_y = ninety_deg_rot @ desired_new_x
desired_delta_pos = desired_forward_distance * desired_new_y
desired_new_pos = init_pos + desired_delta_pos

plot_robot(desired_new_pos, desired_new_x, "red")
plot_robot(desired_new_pos, desired_new_y, "red")

delta_world_position = np.array([-1, 3])
delta_robot_heading = np.deg2rad(10)

new_robot_pos = init_pos + delta_world_position
new_robot_heading = init_heading + delta_robot_heading

new_rot = R.from_euler("z", new_robot_heading, degrees=False)
new_rot_mat = new_rot.as_matrix()[:2, :2]

new_robot_x = new_rot_mat @ np.array([1, 0])
new_robot_y = ninety_deg_rot @ new_robot_x

desired_score = desired_delta_pos @ desired_delta_pos
normalization = min(1.0, np.linalg.norm(desired_delta_pos) / np.linalg.norm(delta_world_position))
score = delta_world_position @ desired_delta_pos * normalization
print(f"Desired Score = {desired_score:.2f}")
print(f"Score = {score:.2f}")

plot_robot(new_robot_pos, new_robot_x, "b")
plot_robot(new_robot_pos, new_robot_y, "b")

plt.xlim(-10, 10)
plt.ylim(-10, 10)

# plt.show()
