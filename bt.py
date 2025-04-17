# move2object_bt.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json

# Append the parent directory to sys.path for relative imports.
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
from classrobot import gripper

##############################################################
# Behavior Tree Node Classes
##############################################################

class BTNode:
    """Base class for behavior tree nodes."""
    def tick(self, context):
        raise NotImplementedError("tick() must be implemented by subclass.")

class Sequence(BTNode):
    """Executes child nodes sequentially until one fails."""
    def __init__(self, children):
        self.children = children

    def tick(self, context):
        for child in self.children:
            if not child.tick(context):
                return False
        return True

class Selector(BTNode):
    """
    Executes child nodes sequentially until one succeeds.
    If none succeed, returns False.
    """
    def __init__(self, children):
        self.children = children

    def tick(self, context):
        for child in self.children:
            if child.tick(context):
                return True
        return False

##############################################################
# Leaf Task Nodes
##############################################################

class MoveHomeTask(BTNode):
    """Task to move the robot to its home position."""
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Move Home")
        self.mover.move_home()
        return True

class GetMarkersTask(BTNode):
    """Task to get markers from camera and transform their coordinates."""
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Get Markers from Camera")
        marker_points = self.mover.cam_relasense()
        if not marker_points:
            print("[ERROR] No markers detected by the camera.")
            return False
        transformed = self.mover.transform_marker_points(marker_points, self.mover.best_matrix)
        if not transformed:
            print("[ERROR] No transformed markers available.")
            return False
        context['transformed_markers'] = transformed
        return True

class CheckPickMarkerTask(BTNode):
    """
    Task to select a marker for picking based on highest z-coordinate.
    The selected marker is saved in the context.
    """
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Check Pick Marker (highest z)")
        markers = context.get('transformed_markers', [])
        if not markers:
            print("[ERROR] No markers in context.")
            return False
        pick_marker = self.mover.check_box_is_stack(markers)
        if not pick_marker:
            print("[ERROR] No suitable pick marker found.")
            return False
        context['pick_marker'] = pick_marker
        return True

class PickBoxTask(BTNode):
    """Task to pick up the box from the selected marker."""
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        pick_marker = context.get('pick_marker', None)
        if not pick_marker:
            print("[ERROR] No pick marker in context.")
            return False
        print(f"[TASK] Pick box from marker ID {pick_marker['id']}")
        self.mover.pick_box(pick_marker)
        return True

class MoveHomeAfterPickTask(BTNode):
    """Task to move home after picking the box."""
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Move Home after Pick")
        self.mover.move_home()
        return True

class GetEmptyMarkerTask(BTNode):
    """
    Task to retrieve an empty marker (IDs 100, 101, 102)
    for placing the box. The selected marker is saved in the context.
    """
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Get Empty Marker")
        markers = context.get('transformed_markers', [])
        empty_marker = self.mover.get_next_empty_marker(markers)
        if not empty_marker:
            print("[ERROR] No empty marker found.")
            return False
        context['empty_marker'] = empty_marker
        return True

class PlaceBoxTask(BTNode):
    """Task to place the box at the empty marker's position."""
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        empty_marker = context.get('empty_marker', None)
        if not empty_marker:
            print("[ERROR] No empty marker in context.")
            return False
        print(f"[TASK] Place box at empty marker ID {empty_marker['id']}")
        # For empty markers, assume we use the camera's y value as-is (or modify as needed)
        self.mover.place_box_at(empty_marker["point"])
        return True

class SortPickAndPlaceTask(BTNode):
    """
    Fallback task that calls the sort_pick_and_place() method from your
    original code if the primary branch fails.
    """
    def __init__(self, mover):
        self.mover = mover

    def tick(self, context):
        print("[TASK] Running fallback: sort_pick_and_place()")
        self.mover.sort_pick_and_place()
        return True

##############################################################
# Move2Object Class (unchanged core functionality)
##############################################################

class Move2Object:
    def __init__(self):
        self.HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568,
                         -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        self.robot_ip = "192.168.200.10"
        self.speed = 0.025
        self.acceleration = 1.0
        # fix y position and roll-pitch-yaw
        self.FIX_Y = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.Test_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

        # Load the transformation matrix from a JSON file.
        self.config_matrix_path = pathlib.Path(__file__).parent / "config/best_matrix.json"
        # Initialize the robot connection.
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)

        self.cam = realsense_cam.RealsenseCam()
        self.best_matrix = self.load_matrix()

        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()
        # Track which empty markers have been used (markers with IDs 100, 101, 102).
        self.used_empty_markers = []

    def init_gripper(self):
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
        time.sleep(0.6)
        print("Testing gripper ...", end="")
        self.close_gripper()
        time.sleep(2)
        self.open_gripper()
        time.sleep(2)

    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        time.sleep(2.6)
        print("Closing gripper...")
        self._GRIPPER_LEFT_.my_hand_close()
        time.sleep(2)

    def open_gripper(self):
        print("Opening gripper...")
        time.sleep(0.6)
        self._GRIPPER_LEFT_.my_hand_open()
        time.sleep(2)

    def cam_relasense(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
        image_marked, point3d = self.cam.get_all_board_pose(aruco_dict)
        if image_marked is not None:
            cv.imshow("Detected Board", image_marked)
            cv.waitKey(5000)
            cv.destroyAllWindows()
        return point3d

    def load_matrix(self):
        try:
            with open(self.config_matrix_path, 'r') as f:
                loaded_data = json.load(f)
                best_matrix = np.array(loaded_data["matrix"])
                return best_matrix
        except FileNotFoundError:
            print("Transformation matrix file not found.")
            return None

    def get_robot_TCP(self):
        pos = self.robot.robot_get_position()
        pos_3d = self.robot.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS, self.speed)

    def transform_marker_points(self, marker_points, transformation_matrix):
        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            # Create homogeneous coordinate (4x1) by appending a 1.
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            transformed_homo = transformation_matrix @ homo_pt
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points

    def check_box_is_stack(self, transformed_points):
        if not transformed_points:
            print("No markers found.")
            return None
        max_marker = transformed_points[0]
        for marker in transformed_points:
            if marker["point"].z > max_marker["point"].z:
                max_marker = marker
        print(f"Marker with highest z: ID {max_marker['id']} with z = {max_marker['point'].z}")
        return max_marker

    def pick_box(self, marker):
        point = marker["point"]
        target_pose_up = [point.x + 0.05, point.y - 0.10, point.z] + self.Test_RPY
        target_pose_down = [point.x + 0.05, point.y, point.z] + self.Test_RPY
        print(f"[PICK] Marker ID {marker['id']}")
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(3)
        self.robot.robot_moveL(target_pose_down, self.speed)
        time.sleep(3)
        self.close_gripper()
        time.sleep(3)
        self.robot.robot_moveL(target_pose_up, self.speed)
        time.sleep(3)

    def find_next_stack_position(self, current_id, sorted_markers):
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None

    def place_box_at(self, point):
        # For placement, compute an approach ("up") pose and a placement ("down") pose.
        stack_pose_up = [point.x, point.y - 0.05, point.z] + self.Test_RPY
        stack_pose_down = [point.x, point.y - 0.1, point.z] + self.Test_RPY
        print(f"[STACK] Placing box at position {point}")
        self.robot.robot_moveL(stack_pose_up, self.speed)
        time.sleep(3)
        self.robot.robot_moveL(stack_pose_down, self.speed)
        time.sleep(3)
        self.open_gripper()
        time.sleep(3)
        self.robot.robot_moveL(stack_pose_up, self.speed)

    def sort_pick_and_place(self):
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

    def get_next_empty_marker(self, transformed_points):
        empty_ids = {100, 101, 102}
        for marker in transformed_points:
            if marker["id"] in empty_ids and marker["id"] not in self.used_empty_markers:
                self.used_empty_markers.append(marker["id"])
                print(f"[EMPTY] Using empty marker ID {marker['id']}")
                return marker
        print("[EMPTY] No available empty marker found.")
        return None

##############################################################
# Main: Build and Run the Behavior Tree
##############################################################

def main():
    mover = Move2Object()
    # Create a shared context dictionary for passing data between nodes.
    context = {}

    # Build primary sequence: move home, get markers, select pick marker, pick box, move home, get empty marker, place box.
    primary_sequence = Sequence([
        MoveHomeTask(mover),
        GetMarkersTask(mover),
        CheckPickMarkerTask(mover),
        PickBoxTask(mover),
        MoveHomeAfterPickTask(mover),
        GetEmptyMarkerTask(mover),
        PlaceBoxTask(mover)
    ])

    # Fallback task: use sort_pick_and_place if primary fails.
    fallback_task = SortPickAndPlaceTask(mover)

    # Top-level tree is a selector: try primary sequence, if it fails, use fallback.
    top_level_tree = Selector([
        primary_sequence,
        fallback_task
    ])

    # Tick (execute) the tree.
    print("[BT] Starting Behavior Tree execution")
    success = top_level_tree.tick(context)
    if success:
        print("[BT] Behavior Tree finished successfully.")
    else:
        print("[BT] Behavior Tree execution failed.")
    # Optionally, you could keep ticking the tree in a loop if your application is continuous.

if __name__ == "__main__":
    main()
