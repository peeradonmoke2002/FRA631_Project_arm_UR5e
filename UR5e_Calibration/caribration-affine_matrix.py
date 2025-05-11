import csv
import numpy as np
import math
import os
import sys
import pathlib
import json

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot.point3d import Point3D
from classrobot.calibrationdata import CalibrationData
from classrobot.caribration_tool import CalibratorTools


# Path to your CSV file
filename = pathlib.Path(__file__).parent.parent / "data" / "calibration_data.csv"
if not os.path.exists(filename):
    raise FileNotFoundError(f"The file '{filename}' does not exist. Please check the path.")

calibration_data_list = []

with open(filename, 'r', newline='', encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile)
    # Strip extra whitespace from header keys
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    print("Detected header keys:", reader.fieldnames)
    
    for row in reader:
        # Ensure the header names match your CSV.
        pos = int(row["Pos"])
        ccs_x = float(row["ccs_x"])
        ccs_y = float(row["ccs_y"])
        ccs_z = float(row["ccs_z"])
        ac_x  = float(row["ac_x"])
        ac_y  = float(row["ac_y"])
        ac_z  = float(row["ac_z"])
        
        # Create Point3D objects for both coordinate systems.
        ccs_point = Point3D(ccs_x, ccs_y, ccs_z)
        ac_point  = Point3D(ac_x, ac_y, ac_z)
        
        # Create a CalibrationData instance and add it to the list.
        calib_data = CalibrationData(pos, ccs_point, ac_point)
        calibration_data_list.append(calib_data)

print(f"Loaded {len(calibration_data_list)} calibration data points.")

# Create an instance of the Calibrator class using the loaded data.
calibrator = CalibratorTools(calibration_data_list)

# Set calibration parameters.
num_selected_positions = len(calibration_data_list)  # using all loaded data points
num_iterations = 50000
target_rms_error = 0.001
# Run the calibration search.
best_matrix, best_rms, rms_errors, selected_positions, transformed_points = calibrator.find_best_matrix(
    num_selected_positions, num_iterations, target_rms_error
)

data = {
    "name": "camera_to_TCP_transform",
    "matrix": best_matrix.tolist()
}

config_path = pathlib.Path(__file__).parent.parent / "configs" / "best_matrix.json"
print(f"Saving best matrix to {config_path}")

# Ensure the parent directory exists (but do not recreate if it already exists).
config_path.parent.mkdir(parents=True, exist_ok=True)

# Write to the file, replacing its contents if it already exists.
with open(config_path, 'w') as f:
    json.dump(data, f, indent=4)

# Output the results.
if best_matrix is not None:
    print("Best Transformation Matrix (4x4):")
    print(best_matrix)
    print("Best Overall RMS Error:", np.round(best_rms, 3))
    print("RMS Errors (X, Y, Z, Overall):", np.round(rms_errors, 3))
    #print("Selected Positions Indices:", selected_positions)
else:
    print("Failed to estimate a valid transformation.")
