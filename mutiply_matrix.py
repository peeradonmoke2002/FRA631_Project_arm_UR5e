import json
import pathlib
import numpy as np

# Define the empty position in Cartesian coordinates.
empty_pos = [0.388, 0.173, 1.509]

# Define the path to the transformation matrix JSON file.
# This is the same relative location used in the original file.
config_matrix_path = pathlib.Path(__file__).parent / "/home/tang/ur_robot/FRA631_Project_Dual_arm_UR5_Calibration/caribration/config/best_matrix.json"

# Load the transformation matrix from the file.
try:
    with open(config_matrix_path, "r") as f:
        loaded_data = json.load(f)
        best_matrix = np.array(loaded_data["matrix"])
except FileNotFoundError:
    print("Transformation matrix file not found.")
    exit()

# Convert the empty position to homogeneous coordinates by appending a 1.
empty_homo = np.array(empty_pos + [1.0], dtype=np.float32).reshape(4, 1)

# Multiply the transformation matrix by the homogeneous empty position.
transformed_empty_space = best_matrix @ empty_homo

# Convert back to Cartesian coordinates by dividing by the homogeneous coordinate.
transformed_empty = (transformed_empty_space[:3, 0] / transformed_empty_space[3, 0]).tolist()

# Print the transformed empty pose.
print("Transformed empty pose:", transformed_empty)
