import realsense_class
import cv2
import json
import numpy as np


with open("realsense_calibration.json", "r") as f:
    calib_data = json.load(f)

# Using the rectified parameters (for example, "rectified.0") to build the camera matrix.
fx = float(calib_data["rectified.0.fx"])
fy = float(calib_data["rectified.0.fy"])
ppx = float(calib_data["rectified.0.ppx"])
ppy = float(calib_data["rectified.0.ppy"])

camera_matrix = np.array([
    [fx, 0, ppx],
    [0, fy, ppy],
    [0,  0,   1]
], dtype=np.float32)

# If your JSON doesn't include distortion coefficients, you can assume zero distortion.
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

print("Loaded Camera Matrix:")
print(camera_matrix)
print("Loaded Distortion Coefficients:")
print(dist_coeffs)

cam = realsense_class.RealsenseCam(width=640, height=480, fps=30)
# Import the RealsenseCam class (assuming it’s in your module)

# Define board parameters
board_size = (5, 5)       # (columns, rows)
square_length = 20        # 20 mm per square, so 5 squares give 100 mm total width/height
marker_length = 15        # Example marker length (15 mm); adjust as needed

# Select the predefined ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Create the ChArUco board instance using the new API
charuco_board = cv2.aruco.CharucoBoard(
    size=board_size,
    squareLength=square_length,
    markerLength=marker_length,
    dictionary=aruco_dict
)

# 4. Estimate the board pose.
image_marked = cam.get_board_pose(charuco_board, camera_matrix, dist_coeffs)


# Display the marked image if available.
if image_marked is not None:
    cv2.imshow("Detected Board", image_marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. Stop the camera.
# cam.stop()





# 5. Stop the camera.
cam.stop()
