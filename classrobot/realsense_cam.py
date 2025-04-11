#!/usr/bin/env python3
"""
move2object.py

This module controls a simulated robot and gripper to pick and place objects 
based on transformed marker points. The module loads a calibration matrix,
obtains marker positions (from test data in this simulation), transforms them, 
and uses a Behavior Tree to process overlapped markers first and then sorted markers.
The simulated robot “moves” by printing commands.
  
Author: Your Name
Date: YYYY-MM-DD
"""

import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json
import math
from scipy.spatial.transform import Rotation as R

# Add parent directory to sys.path so modules can be imported
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from classrobot import robot_movement, realsense_cam, gripper
from classrobot.point3d import Point3D


# ========================================
# Behavior Tree Framework (Simple Implementation)
# ========================================

SUCCESS = 1
FAILURE = 0
RUNNING = 2

class BehaviorNode:
    def tick(self):
        raise NotImplementedError("tick() must be implemented by subclass.")

class SequenceNode(BehaviorNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != SUCCESS:
                return status
        return SUCCESS

class SelectorNode(BehaviorNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status == SUCCESS:
                return SUCCESS
        return FAILURE

class ConditionNode(BehaviorNode):
    def __init__(self, condition_func):
        self.condition_func = condition_func

    def tick(self):
        if self.condition_func():
            return SUCCESS
        else:
            return FAILURE

class ActionNode(BehaviorNode):
    def __init__(self, action_func):
        self.action_func = action_func

    def tick(self):
        return self.action_func()


# ========================================
# Simulated Move2Object Class with Behavior Tree Integration
# ========================================

class Move2Object:
    """
    Class to move the robot to pick and place objects based on detected markers.
    """
    def __init__(self):
        # Define home position and other fixed parameters
        self.HOME_POS = [
            0.701172053107018, 0.184272460738082, 0.1721568294843568,
            -1.7318488600590023, 0.686830145115122, -1.731258978679887
        ]
        self.robot_ip = "172.17.0.2"
        self.speed = 0.1
        self.acceleration = 1.2

        # Fixed Y value and test roll-pitch-yaw for object approach
        self.FIX_Y = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.Test_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

        # Overlap threshold (adjustable)
        self.overlap_threshold = 0.05

        # Load transformation matrix from JSON calibration file
        self.config_matrix_path = pathlib.Path(__file__).parent / (
            "FRA631_Project_Dual_arm_UR5_Calibration/caribration/config/best_matrix.json"
        )
        self.best_matrix = self.load_matrix()

        # Initialize robot and gripper connections (simulation version)
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)
        self.cam = realsense_cam.RealsenseCam()
        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()

        # Build the behavior tree.
        self.build_behavior_tree()

    def init_gripper(self):
        """
        Initialize and test the gripper.
        """
        host = "192.168.200.11"  # Replace with your gripper's IP address
        port = 502  # Default Modbus TCP port
        print(f"Connecting to 3-Finger gripper at {host}:{port}...", end=" ")
        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()
        time.sleep(0.6)
        print("Testing gripper...", end=" ")
        self.close_gripper()
        time.sleep(2)
        self.open_gripper()
        time.sleep(2)

    def stop_all(self):
        """Release robot, camera, and gripper resources."""
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        """Close the gripper."""
        time.sleep(0.6)
        print("Closing gripper...")
        self._GRIPPER_LEFT_.my_hand_close()
        time.sleep(2)

    def open_gripper(self):
        """Open the gripper."""
        print("Opening gripper...")
        time.sleep(0.6)
        self._GRIPPER_LEFT_.my_hand_open()
        time.sleep(2)

    def cam_realsense(self):
        """
        Capture board pose using the RealSense camera.
        In the real system, this would capture and detect markers.
        Here it is meant to be overridden in simulation.
        Returns:
            List of marker dictionaries with keys "id" and "point" (Point3D).
        """
        raise NotImplementedError("cam_realsense() should be overridden for simulation.")

    def load_matrix(self):
        """
        Load the transformation matrix from a JSON file.
        Returns:
            best_matrix (numpy.array): 4x4 transformation matrix.
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
        Get current robot TCP (Tool Center Point) position.
        Returns:
            pos_3d (list): [x, y, z] position.
        """
        pos = self.robot.robot_get_position()
        pos_3d = self.robot.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        """Move the robot to its home position."""
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS, self.speed)

    def transform_marker_points(self, marker_points, transformation_matrix):
        """
        Apply a transformation matrix to a list of marker points.
        
        Args:
            marker_points (list): List of dicts, each in the format {"id": <marker_id>, "point": Point3D}.
            transformation_matrix (numpy.array): 4x4 transformation matrix.
        
        Returns:
            transformed_points (list): List of dicts with transformed "point" as a Point3D.
        """
        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            transformed_homo = transformation_matrix @ homo_pt
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points

    def reorder_markers_for_overlap(self, markers, threshold=None, z_threshold=0.05):
        """
        Reorder markers so that overlapping markers are processed first.
        In this modified version, if the difference in the z coordinate between two markers
        is less than `z_threshold`, they are considered overlapping immediately.
        
        Args:
            markers (list): List of dicts, each with keys "id" and "point" (Point3D).
            threshold (float, optional): A fallback distance threshold to decide overlap.
                                         Defaults to self.overlap_threshold.
            z_threshold (float, optional): Threshold for the difference in z coordinates.
                                           Defaults to 0.05.
        
        Returns:
            ordered_markers (list): Reordered list of markers.
        """
        if threshold is None:
            threshold = self.overlap_threshold

        n = len(markers)
        overlap_flags = [False] * n

        # Check overlap using only the z coordinate.
        for i in range(n):
            for j in range(i + 1, n):
                pt1 = markers[i]["point"]
                pt2 = markers[j]["point"]
                if abs(pt1.z - pt2.z) < z_threshold:
                    overlap_flags[i] = True
                    overlap_flags[j] = True

        overlapped = [markers[i] for i in range(n) if overlap_flags[i]]
        non_overlapped = [markers[i] for i in range(n) if not overlap_flags[i]]

        overlapped_sorted = sorted(overlapped, key=lambda m: m["point"].y)
        non_overlapped_sorted = sorted(non_overlapped, key=lambda m: m["id"])

        return overlapped_sorted + non_overlapped_sorted

    def get_overlapped_markers(self, markers, z_threshold=None):
        """
        Returns a list of markers that are considered overlapped based solely on their z coordinate.
        Two markers are considered overlapped if the absolute difference in z is below z_threshold.
        
        Args:
            markers (list): List of dicts, each with keys "id" and "point" (Point3D).
            z_threshold (float, optional): The z threshold for overlap. Defaults to self.overlap_threshold.
        
        Returns:
            overlapped (list): List of markers considered as overlapped.
        """
        if z_threshold is None:
            z_threshold = self.overlap_threshold

        overlapped = []
        n = len(markers)
        for i in range(n):
            for j in range(i + 1, n):
                pt1 = markers[i]["point"]
                pt2 = markers[j]["point"]
                if abs(pt1.z - pt2.z) < z_threshold:
                    if markers[i] not in overlapped:
                        overlapped.append(markers[i])
                    if markers[j] not in overlapped:
                        overlapped.append(markers[j])
        return overlapped

    def process_marker(self, marker):
        """
        Process one marker: compute target poses, move to pick and then place the object.
        """
        marker_id = marker["id"]
        point = marker["point"]

        # Calculate target poses (above the object and at the object).
        target_pose_up = [point.x + 0.05, point.y - 0.10, point.z] + self.Test_RPY
        target_pose_down = [point.x + 0.05, point.y, point.z] + self.Test_RPY

        print(f"Moving to marker ID {marker_id} at position {target_pose_up}")
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(3)
        self.robot.robot_moveL(target_pose_down, self.speed)
        # Here, in simulation, you might disable gripper commands:
        # self.close_gripper()
        time.sleep(3)
        self.move_home()
        time.sleep(3)
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(3)
        self.robot.robot_moveL(target_pose_down, self.speed)
        # self.open_gripper()
        time.sleep(3)
        self.move_home()
        time.sleep(3)
        print(f"Completed move for marker ID {marker_id}")

    def sort_and_stack_markers(self, markers):
        """
        Sort markers by id (ascending), and then process them sequentially.
        This is used when there is no overlapping detected.
        """
        sorted_markers = sorted(markers, key=lambda m: m["id"])
        for marker in sorted_markers:
            self.process_marker(marker)

    def build_behavior_tree(self):
        """
        Build the behavior tree to control the pick-and-place operations.
        
        Tree Structure:
                        Selector
                          /    \
                [Overlap Sequence]   [Sorted Sequence]
                     |                         |
         [Check Overlap Condition]       [Process Sorted Action]

        If overlapped markers are detected, process them first;
        otherwise, sort all markers by id and process them.
        """
        self.bt_check_overlap = ConditionNode(self.bt_condition_overlap)
        self.bt_process_overlapped = ActionNode(self.bt_action_process_overlapped)
        self.bt_process_sorted = ActionNode(self.bt_action_process_sorted)

        self.bt_sequence_overlap = SequenceNode([self.bt_check_overlap, self.bt_process_overlapped])
        self.bt_sequence_sorted = SequenceNode([self.bt_process_sorted])
        self.bt_root = SelectorNode([self.bt_sequence_overlap, self.bt_sequence_sorted])

    def bt_condition_overlap(self):
        """
        Condition Node: Checks if there are overlapped markers.
        Returns SUCCESS if overlapped markers exist; otherwise, FAILURE.
        """
        marker_points = self.cam_realsense()
        if marker_points is None:
            print("No marker points received.")
            return False
        transformed = self.transform_marker_points(marker_points, self.best_matrix)
        self.bt_all_markers = sorted(transformed, key=lambda m: m["id"])
        self.bt_overlapped = self.get_overlapped_markers(self.bt_all_markers)
        if len(self.bt_overlapped) > 0:
            print(f"Detected {len(self.bt_overlapped)} overlapped marker(s).")
            return True
        else:
            print("No overlapped markers detected.")
            return False

    def bt_action_process_overlapped(self):
        """
        Action Node: Process overlapped markers.
        """
        print("Processing overlapped markers first (custom placement)...")
        for marker in self.bt_overlapped:
            self.process_marker(marker)
        return SUCCESS

    def bt_action_process_sorted(self):
        """
        Action Node: Process markers sorted by id.
        """
        print("Processing markers by sorting them (no overlaps detected).")
        self.sort_and_stack_markers(self.bt_all_markers)
        return SUCCESS

    def run_behavior_tree(self):
        """
        Run the behavior tree by ticking the root node.
        """
        status = self.bt_root.tick()
        if status == SUCCESS:
            print("Behavior tree completed successfully.")
        else:
            print("Behavior tree failed or is running.")

    def move_muti_to_object(self):
        """
        Instead of directly processing markers, run the behavior tree.
        """
        if self.best_matrix is None:
            print("Failed to load transformation matrix.")
            return
        self.run_behavior_tree()


# ---------------------------------------------------
# Simulated Class: Override cam_realsense() and Movement
# ---------------------------------------------------

class SimulatedMove2Object(Move2Object):
    def cam_realsense(self):
        """
        Instead of capturing from a camera, return test marker data.
        Each marker is a dict with "id" and "point" (Point3D) constructed
        from the provided test data.
        """
        test_data = [
            {"id": 1, "x": 0.47737592458724976, "y": 0.05334507301449776, "z": -1.4030000400543213},
            {"id": 2, "x": -0.22360965609550476, "y": -0.0995423257350922, "z": -1.4980000257492065},
            {"id": 3, "x": 0.011661688797175884, "y": -0.10243550688028336, "z": -1.4980000257492065},
            {"id": 4, "x": 0.2854086458683014, "y": 0.00665783792734146, "z": -1.505000114440918},
            {"id": 5, "x": -0.19570325314998627, "y": 0.07943285256624222, "z": -1.5120000839233398},
            {"id": 6, "x": 0.021726226434111595, "y": 0.05677981302142143, "z": -1.5090000629425049}
        ]
        markers = []
        for item in test_data:
            pt = Point3D(item["x"], item["y"], item["z"])
            markers.append({"id": item["id"], "point": pt})
        print("Simulated camera detects the following markers:")
        for marker in markers:
            pt = marker["point"]
            print(f"  Marker ID {marker['id']} at ({pt.x:.3f}, {pt.y:.3f}, {pt.z:.3f})")
        return markers

    def __init__(self):
        super().__init__()
        # Override movement methods to simulate (print) actions.
        self.robot.robot_moveL = lambda pose, speed: print(f"Simulated moveL to pose: {pose}, speed: {speed}")
        self.move_home = lambda: print("Simulated move_home called")
        self.stop_all = lambda: print("Simulated stop_all called")


def main():
    # Create an instance of the simulated class.
    move_object = SimulatedMove2Object()
    move_object.move_home()
    time.sleep(2)
    move_object.move_muti_to_object()
    time.sleep(5)  # Wait before returning home
    move_object.move_home()
    move_object.stop_all()


if __name__ == "__main__":
    main()
