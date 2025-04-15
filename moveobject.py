# move2object.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json

# Append the parent directory to sys.path to allow relative module imports.
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
from classrobot import gripper


class Move2Object:
    def __init__(self):
        self.HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568,
                         -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        self.robot_ip = "192.168.200.10"
        self.speed = 0.02
        self.acceleration = 1.0
        # fix y position and roll-pitch-yaw
        self.FIX_Y = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.Test_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

        # Load the transformation matrix from a JSON file.
        self.config_matrix_path = pathlib.Path(__file__).parent / "FRA631_Project_Dual_arm_UR5_Calibration/caribration/config/best_matrix.json"
        # Initialize the robot connection once.
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)

        self.cam = realsense_cam.RealsenseCam()
        self.empty_pos = [0.388, 0.173, 1.509]
        self.best_matrix = self.load_matrix()

        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()

    def init_gripper(self):
        # Initialize the gripper connection.
        host = "192.168.200.11"  # Replace with your gripper's IP address.
        port = 502               # Typically the default Modbus TCP port.
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()

        time.sleep(0.6)  # Delay slightly longer than the TIME_PROTECTION (0.5 s).
        print("Testing gripper ...", end="")
        self.close_gripper()  # Test the close command.
        time.sleep(2)
        self.open_gripper()   # Test the open command.
        time.sleep(2)

    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        """
        Closes the gripper.
        """
        time.sleep(0.6)
        print("Closing gripper...")
        self._GRIPPER_LEFT_.my_hand_close()
        time.sleep(2)

    def open_gripper(self):
        """
        Opens the gripper.
        """
        print("Opening gripper...")
        time.sleep(0.6)
        self._GRIPPER_LEFT_.my_hand_open()
        time.sleep(2)

    def cam_relasense(self):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose as a list of dictionaries in the format:
            [{"id": marker_id, "point": Point3D(x, y, z)}, ...]
        """
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
        image_marked, point3d = self.cam.get_all_board_pose(aruco_dict)
        if image_marked is not None:
            cv.imshow("Detected Board", image_marked)
            cv.waitKey(5000)
            cv.destroyAllWindows()
        return point3d

    def load_matrix(self):
        """
        Loads the transformation matrix from a file.
        Returns the transformation matrix (numpy array) or None if not found.
        """
        try:
            with open(self.config_matrix_path, 'r') as f:
                loaded_data = json.load(f)
                best_matrix = np.array(loaded_data["matrix"])
                return best_matrix
        except FileNotFoundError:
            print("Transformation matrix file not found.")
            return None

    def get_robot_TCP(self):
        """
        Connects to the robot and retrieves the current TCP (end-effector) position.
        Returns a 3-element list: [x, y, z].
        """
        pos = self.robot.robot_get_position()
        pos_3d = self.robot.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS, self.speed)

    def transform_marker_points(self, marker_points, transformation_matrix):
        """
        Applies a 4x4 transformation matrix to a list of marker points.
        Returns a list of dictionaries with the transformed point and the marker id.
        """
        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            # Create homogeneous coordinate (4x1) by appending a 1.
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            # Multiply by the transformation matrix.
            transformed_homo = transformation_matrix @ homo_pt
            # Convert back to Cartesian coordinates.
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points

    def empyty_pose(self):
        """
        Transforms the stored empty position to the robot base frame.
        Returns the empty position as a list: [x, y, z, roll, pitch, yaw].
        """
        empty_homo = np.array(self.empty_pos + [1.0], dtype=np.float32).reshape(4, 1)
        transformed_empty_space = self.best_matrix @ empty_homo
        transformed_empty = (transformed_empty_space[:3, 0] /
                             transformed_empty_space[3, 0]).tolist()
        return transformed_empty 


    def check_box_is_stack(self, transformed_points):
        """
        Checks if any box is stacked based on the z-coordinate of the transformed points.
        Returns True if a marker has a z-coordinate below 3.6, False otherwise.
        """
        for marker in transformed_points:
            point = marker["point"]
            if point.z < 3.6:
                print("Box is stacked.")
                return True
        print("Box is not stacked.")
        return False

    def pick_box(self, marker):
        """Move above and pick the box at the marker position."""
        point = marker["point"]
        target_pose_up = [point.x + 0.05, point.y - 0.10, point.z + 0.1] + self.Test_RPY
        target_pose_down = [point.x + 0.05, point.y, point.z] + self.Test_RPY

        print(f"[PICK] Marker ID {marker['id']}")
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(1)
        self.robot.robot_moveL(target_pose_down, self.speed)
        time.sleep(1)
        self.close_gripper()
        time.sleep(1)
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(1)

    def find_next_stack_position(self, current_id, sorted_markers):
        """Finds the next marker that has a higher ID."""
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None

    def place_box_at(self, point):
        """Place the box at the specified 3D point."""
        stack_pose_up = [point.x, point.y-10, point.z] + self.Test_RPY
        stack_pose_down = [point.x, point.y, point.z] + self.Test_RPY

        print(f"[STACK] Placing box at position {point}")
        self.robot.robot_moveL(stack_pose_up, self.speed)
        time.sleep(1)
        self.robot.robot_moveL(stack_pose_down, self.speed)
        time.sleep(1)
        self.open_gripper()
        time.sleep(1)
        self.robot.robot_moveL(stack_pose_up, self.speed)
        time.sleep(1)

    def sort_pick_and_place(self):
        """
        Main function: pick boxes by sorted ID and stack them at the next higher marker position.
        """
        if self.best_matrix is None:
            print("Failed to load transformation matrix.")
            return

        marker_points = self.cam_relasense()
        transformed_points = self.transform_marker_points(marker_points, self.best_matrix)
        sorted_markers = sorted(transformed_points, key=lambda m: m["id"])

        for marker in sorted_markers:
            current_id = marker["id"]
            self.pick_box(marker)
            next_marker = self.find_next_stack_position(current_id, sorted_markers)
            if next_marker:
                self.place_box_at(next_marker["point"])
            else:
                print(f"[SKIP STACK] No higher marker found for marker ID {current_id}")


def main():
    move2object = Move2Object()
    move2object.move_home()
    time.sleep(2)

    # Retrieve marker points and transform them.
    marker_points = move2object.cam_relasense()
    transformed_points = move2object.transform_marker_points(marker_points, move2object.best_matrix)

    # Check if a box is  stacked.
    if move2object.check_box_is_stack(transformed_points):
        print("[INFO] A box is  stacked.")
        # For example, pick the first stacked box (marker with z < 3.6).
        for marker in transformed_points:
            if marker["point"].z < 3.6:
                move2object.pick_box(marker)
                break
        # Get the empty position and place the box there.
        empty_pos = move2object.empty_pos()
        empty_point = Point3D(empty_pos[0], empty_pos[1], empty_pos[2])
        move2object.place_box_at(empty_point)
    else:
        move2object.sort_pick_and_place()


if __name__ == "__main__":
    main()
