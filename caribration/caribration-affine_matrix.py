import numpy as np
import math
from typing import List, Tuple, Optional
from caribration_utility import Points3D, CalibrationData, Calibrator

# Create a known transformation for synthetic testing.
angle = math.radians(15)  # 15 degree rotation about Z axis
R = np.array([[math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle),  math.cos(angle), 0],
                [0,                0,               1]], dtype=np.float32)
t = np.array([10, 20, 30], dtype=np.float32)
true_affine = np.hstack([R, t.reshape(3, 1)])  # 3x4 matrix
true_homogeneous = np.vstack([true_affine, np.array([0, 0, 0, 1], dtype=np.float32)])

# Generate synthetic calibration data.
calibration_data_list = []
num_points = 36
np.random.seed(42)
for i in range(num_points):
    # Generate a random source (CCS) point.
    x, y, z = np.random.uniform(0, 100, 3)
    source_pt = Points3D(x, y, z)
    # Transform the source point to get the target (AC) point.
    homo_pt = np.array([x, y, z, 1], dtype=np.float32)
    transformed = true_homogeneous @ homo_pt
    target_pt = Points3D(transformed[0], transformed[1], transformed[2])
    # Create a CalibrationData instance.
    calib_data = CalibrationData(pos=i, CCS=source_pt, AC=target_pt)
    calibration_data_list.append(calib_data)

# Create an instance of the Calibrator class.
calibrator = Calibrator(calibration_data_list)

# Run the calibration search.
num_selected_positions = 36  # Minimal number needed for affine estimation.
num_iterations = 20000
target_rms_error = 0.1  # Set your desired threshold.

best_matrix, best_rms, rms_errors, selected_positions, transformed_points = calibrator.find_best_matrix(
    num_selected_positions, num_iterations, target_rms_error
)

if best_matrix is not None:
    print("Best Transformation Matrix (4x4):")
    print(best_matrix)
    print("Best Overall RMS Error:", np.round(best_rms,3))
    print("RMS Errors (X, Y, Z, Overall):", np.round(rms_errors,3))
    print("Selected Positions Indices:", selected_positions)
else:
    print("Failed to estimate a valid transformation.")
