
        #self.empty_pos = [0.388, 0.173, 1.509]
        #self.safe_pos_right_hand = [-0.46091229133782063, -0.03561778716957069, 0.5597582676128492, -0.026107352593298303, -1.6156259386208272, 0.011223536317220348]
        #self.HOME_POS_right_hand =    [-0.7000166613125681, 0.17996458960475226, 0.17004862107257468, -0.014724582993124808, -1.5742326761027705, -0.016407326458784333]   
        #self.HOME_JPOS = [  ]


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
        self.robot_left_ip = "192.168.200.10"
        self.robot_right_ip = "192.168.200.10"
        self.speed = 0.025
        self.acceleration = 1.0
        self.FIX_Y_LEFT = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.Test_RPY_LEFT = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
        self.config_matrix_path = pathlib.Path(__file__).parent / "config/best_matrix.json"

        self.robot_left = robot_movement.RobotControl()
        self.robot_right = robot_movement.RobotControl()

        self.robot_left.robot_release()
        self.robot_right.robot_release()

        self.robot_left.robot_init(self.robot_left_ip)
        self.robot_right.robot_init(self.robot_right_ip)

        self.cam = realsense_cam.RealsenseCam()
 
        self.recive_pos = [ ]
        
        self.best_matrix = self.load_matrix()
    
        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()
        # Track which empty markers have been used (markers with IDs 100, 101, 102).
        self.used_empty_markers = []
        

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

        time.sleep(0.6)  # Slight delay.
        print("Testing gripper ...", end="")
        self.close_gripper()  # Test the close command.
        time.sleep(2)
        self.open_gripper()   # Test the open command.
        time.sleep(2)

    def stop_all(self):
        self.robot_left.robot_release()
        self.robot_right.robot_release()
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
        Connects to the robot_left and retrieves the current TCP (end-effector) position.
        Returns a 3-element list: [x, y, z].
        """
        pos = self.robot_left.robot_get_position()
        pos_3d = self.robot_left.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        print("Moving to home position...")
        self.robot_left.robot_moveL(self.HOME_POS, self.speed)

    def ensure_home_position(self, tol=1e-3):
        """
        Check current joint angles against HOME_JPOS.
        If any joint differs by more than tol, perform a joint move to HOME_JPOS.
        """
        # Retrieve current joint positions (6-element list)
        current_joints = self.robot_left.robot_get_actual_joint_positions()
        # Compute absolute differences
        diffs = [abs(c - h) for c, h in zip(current_joints, self.HOME_JPOS)]
        if any(d > tol for d in diffs):
            print("Joints not at home. Moving to HOME_JPOS:", self.HOME_JPOS)
            # Joint-space move
            self.robot_left.robot_moveJ(self.HOME_JPOS, self.speed, self.acceleration)
        else:
            print("Already at HOME_JPOS. No movement needed.")


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

    def min_marker(self, transformed_points):
        """
        Find the marker with the lowest Z *only* among real boxes (ID < 100).
        """
        # keep only real‑box IDs
        candidates = [m for m in transformed_points if m["id"] < 100]
        if not candidates:
            print("No box markers found.")
            return None

        # pick the one with min Z
        bottom = min(candidates, key=lambda m: m["point"].y)
        print(f"Marker with lowest z: ID {bottom['id']} (z={bottom['point'].y})")
        return bottom
    

    def max_marker(self, transformed_points):
        """
        Find the marker with the highest Z *only* among real boxes (ID < 100).
        """
        # keep only real-box IDs
        candidates = [m for m in transformed_points if m["id"] < 100]
        if not candidates:
            print("No box markers found.")
            return None

        top = max(candidates, key=lambda m: m["point"].y)
        print(f"Marker with highest z: ID {top['id']} (z={top['point'].y})")
        return top

    def pick_box(self, marker):
        """
        1) Approach over the box: move in X–Z, keep Y fixed.
        2) Descend to the actual Y of the box and close gripper.
        3) Retract back to the approach pose.
        """
        p = marker["point"]
     
        approach = [p.x + 0.05, self.FIX_Y_LEFT, p.z] + self.Test_RPY_LEFT
        pick_pos = [p.x + 0.05, p.y,p.z] + self.Test_RPY_LEFT

        print(f"[PICK] Marker ID {marker['id']} at (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot_left.robot_moveL(approach, self.speed)
        time.sleep(3)
        # descend in Y
        self.robot_left.robot_moveL(pick_pos, self.speed)
        # grip
        self.close_gripper()
        time.sleep(3)
        # retract back to approach pose
        self.robot_left.robot_moveL(approach, self.speed)
        time.sleep(3)

    def place_box(self, marker_point):
        """
        1) Approach over the place point in X–Z, Y fixed.
        2) Descend to actual Y of the place marker and open gripper.
        3) Retract back to the approach pose.
        """
        p = marker_point
        # approach pose (X–Z move, Y fixed)
        approach = [p.x, self.FIX_Y_LEFT, p.z] + self.Test_RPY_LEFT
        # actual place pose
        place_pos = [p.x, p.y-0.077,p.z] + self.Test_RPY_LEFT

        print(f"[PLACE] at marker pos (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot_left.robot_moveL(approach, self.speed)
        time.sleep(1)
        # descend in Y
        self.robot_left.robot_moveL(place_pos, self.speed)
        time.sleep(3)
        # release
        self.open_gripper()
        time.sleep(1)
        # retract
        self.robot_left.robot_moveL(approach, self.speed)
        time.sleep(1)

    def find_next_stack_position(self, current_id, sorted_markers):
        """Finds the next marker that has a higher ID."""
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None


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
                self.place_box(next_marker["point"])
            else:
                print(f"[SKIP STACK] No higher marker found for marker ID {current_id}")

    def get_next_empty_marker(self, transformed_points):
        """
        Searches for an available empty marker among those with IDs 100, 101, or 102.
        If a marker has already been used for placement, it is skipped.
        Returns the marker dictionary if found; otherwise, returns None.
        """
        empty_ids = {100, 101, 102, 103}
        for marker in transformed_points:
            if marker["id"] in empty_ids and marker["id"] not in self.used_empty_markers:
                self.used_empty_markers.append(marker["id"])
                print(f"[EMPTY] Using empty marker ID {marker['id']}")
                return marker
        print("[EMPTY] No available empty marker found.")
        return None
    
    def stack_chain_boxes(self, target_id=104, count=3):
        """
        Stack the lowest-ID boxes in sequence:
        1) First box goes to the position of marker `target_id`.
        2) Each subsequent box goes to the original position of the previously stacked box’s marker.
        """
        raw_pts     = self.cam_relasense()
        transformed = self.transform_marker_points(raw_pts, self.best_matrix)
        # Build a map from marker ID → Point3D
        point_map = {m['id']: m['point'] for m in transformed}

        # Gather all box markers (IDs <100), sort by ascending ID, and take the first `count`
        box_markers = sorted([m for m in transformed if m['id'] < 100],
                             key=lambda m: m['id'])[:count]
        # Chain‐stack 
        prev_id = target_id
        for marker in box_markers:
            box_id = marker['id']

            # Make sure we know the placement point
            if prev_id not in point_map:
                print(f"[STACK] Cannot find position for marker {prev_id}, abort chain.")
                return

            place_pt = point_map[prev_id]
            print(f"[STACK] Picking box {box_id} → placing at marker {prev_id} position")
            self.pick_box(marker)
            self.place_box(place_pt)
            self.move_home()
            time.sleep(1)
            #Now the next box uses this box’s original marker ID
            prev_id = box_id 

    def detect_overlaps(self, pts):
        """
        Group real boxes (ID < 100) whose Y‑coordinate difference is < 0.3 m.
        Returns a list of groups; each group is a list of markers considered
        to be in the same stack purely by their Y proximity.
        """
        STACK_Y_DIFF = 0.3

        # 1) filter down to just the real boxes
        boxes = [m for m in pts if m["id"] < 100]

        groups = []
        used = set()

        for i, base in enumerate(boxes):
            if i in used:
                continue

            group = [base]
            by = base["point"].y

            for j, other in enumerate(boxes[i+1:], start=i+1):
                if j in used:
                    continue

                oy = other["point"].y
                if abs(by - oy) < STACK_Y_DIFF:
                    group.append(other)
                    used.add(j)

            if len(group) > 1:
                used.add(i)
                groups.append(group)

        return groups

    

def main():
    mover = Move2Object()
    mover.move_home()
    time.sleep(2)
    mover.ensure_home_position()
    time.sleep(2)

    # ----- Phase 1: Destack onto markers 100,101,102 only -----
    while True:
        raw_pts     = mover.cam_relasense()
        transformed = mover.transform_marker_points(raw_pts, mover.best_matrix)

        # detect any stacks using the fixed 0.3 m Y rule
        overlaps = mover.detect_overlaps(transformed)   # <-- no y_tol argument
        if not overlaps:
            print("[DESTACK] No more stacks. Moving to arrange phase.")
            break

        # pick lowest Y box
        lowest = mover.min_marker(transformed) 
        if lowest is None:
            print("[DESTACK] No boxes found. Aborting.")
            return

        # deposit onto next empty among 100–102
        candidates = [
            m for m in transformed
            if m["id"] in {100,101,102,103}
            and m["id"] not in mover.used_empty_markers
        ]
        if not candidates:
            print("[DESTACK] No empty destack spots (100–102) left. Aborting.")
            return

        spot = candidates[0]
        mover.used_empty_markers.append(spot["id"])
        print(f"[DESTACK] Moving box {lowest['id']} → marker {spot['id']}")
        mover.pick_box(lowest)
        mover.place_box(spot["point"])
        mover.move_home()
        time.sleep(1)

    # ----- Phase 2: Arrange remaining boxes at marker 104 -----
    print("[ARRANGE] Chain‑stacking final boxes at marker 104")
    mover.stack_chain_boxes(target_id=104, count=3)

    # ----- Cleanup -----
    mover.move_home()
    time.sleep(1)
    mover.stop_all()
    print("Program completed. Robot and camera released.")

if __name__ == "__main__":
    main()


