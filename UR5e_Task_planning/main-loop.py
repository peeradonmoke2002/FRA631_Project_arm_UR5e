import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json
from spatialmath import SE3


# Append the parent directory to sys.path to allow relative module imports.
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot import gripper
from classrobot.UR5e_DH import UR5eDH

PATH_IMAGE_LOGS =  pathlib.Path(__file__).resolve().parent.parent / "images/task_planning/logs"




GRAP_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
GRAP_RPY_HAND = [-2.293352137967249, 0.8661607309064375, 0.03319413211288414]

RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]

HOME_POS_GRAP = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7224438319206319, 0.13545161633255984, -1.2975236351897372] 
 
GRIPPER_HAND_OVER_POS = [0.7011622759203622, 0.18429544171620266, 0.17217042311559283,
                        -2.2933253599444634, 0.8661720762272863, 0.03321393013524738]


HANE_OVER_POS = [-0.5844968810852482, 0.15948833780927751, 0.14179972384122283, 
                 -0.011669515753328431, -1.5741801093628152, -0.019718763405037438]
class TaskPlanning:

    def __init__(self, robot_gripper_ip="192.168.200.10", robot_hand_ip="192.168.200.20"):

        self.HOME_POS = [0.5801247122956066, 0.1840241751374821, 0.38151474428215354, -1.73187267065453,
                          0.6868481312281425, -1.731285115641666]
        
        self.HOME_POS_GRAP = [0.5801247122956066, 0.1840241751374821, 0.38151474428215354, 
            -1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
         
        self.HAND_SAFE_POS=  [-0.5797913816371965, 0.10945501736192521, 0.5868975017296948, 
                 -0.01168892556948376, -1.5742505612720608, -0.019596034953188405]
        self.HAND_OVER_POS = [-0.6960388998978234, 0.22728512918295887, 0.13792559506438679, 
                             -0.0060578982741073, -1.575294320701689, -0.012040935748146972]
        self.GRIPPER_HAND_OVER_POS = [0.49376288336678675, 0.18429329032065667, 0.17215325979069604, 
                                      -2.293352137967249, 0.8661607309064375, 0.03319413211288414]
        self.hand_drop_pos = [-0.5797889893318314, 0.10942745332216032, 0.586847535496484,
                               1.1124348382650098, -1.279905389196877, -1.1336271607829078]
        self.robot_gripper_ip = robot_gripper_ip
        self.robot_hand_ip = robot_hand_ip
        self.speed = 0.03
        self.acceleration = 0.20
        self.FIX_Y = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.medium_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
        self.high_RPY =  [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

        # Create separate controllers for each robot
        self.robot_gripper = robot_movement.RobotControl()
        self.robot_hand    = robot_movement.RobotControl()

        # Release any previous sessions
        self.robot_gripper.robot_release()
        self.robot_hand.robot_release()

        # Initialize both robots
        self.robot_gripper.robot_init(self.robot_gripper_ip)
        self.robot_hand.robot_init(self.robot_hand_ip)

        self.cam = realsense_cam.RealsenseCam()
        self.dt = 0.01

        self.recive_pos = [ ]
        self.getrobotDH()

        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()
        # Track which empty markers have been used (markers with IDs 100, 101, 102).
        self.used_empty_markers = []
        self.group_left = set(range(100, 103))
        self.group_right = set(range(104, 107))
        

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
        self.robot_gripper.robot_release()
        self.robot_hand.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        gr = gripper.MyGripper3Finger()
        # Initialize the gripper
        gr.my_init(host="192.168.200.11", port=502)
        time.sleep(0.6)
        # Test closing
        gr.my_hand_close()
        time.sleep(2)
      
        

    def open_gripper(self):
        gr = gripper.MyGripper3Finger()
        # Initialize the gripper
        gr.my_init(host="192.168.200.11", port=502)
        time.sleep(0.6)
        # Test closing
        gr.my_hand_open()
        time.sleep(2)

    def cam_relasense(self):
        aruco_dict_type = cv.aruco.DICT_5X5_1000
        point3d, images = self.cam.cam_capture_marker_v2(aruco_dict_type)
        # self.save_images(images)
        return point3d

    def save_images(self,image):
        self.cam.save_image(image, PATH_IMAGE_LOGS)

    def get_robot_TCP(self, which='gripper'):
        """
        Connects to the specified robot and retrieves the current TCP (end-effector) position.
        which: 'gripper' or 'hand'
        Returns a 3-element list: [x, y, z].
        """
        if which == 'hand':
            pos = self.robot_hand.robot_get_position()
        else:
            pos = self.robot_gripper.robot_get_position()
        print("Robot TCP position:", pos)
        return pos

    def move_home_gripper(self):
        print("gripper Moving to home position...")
        self.robot_gripper.robot_moveL(self.HOME_POS, self.speed)
    
    def move_home_hand(self):
        print("hand Moving to home position...")
        self.robot_hand.robot_moveL(self.HAND_SAFE_POS, self.speed)

    def robot_hand_over(self):
        print("hand Moving to handover position...")
        self.robot_hand.robot_moveL(self.HAND_OVER_POS, self.speed)

    def robot_gripper_move_home_rpy(self):
        print("gripper Moving to home position...")
        self.robot_gripper.robot_moveL(self.HOME_POS_GRAP, self.speed)
    
    def robot_gripper_position_for_hand_over(self):
        print("gripper Moving to handover position...")
        self.robot_gripper.robot_moveL(self.GRIPPER_HAND_OVER_POS, self.speed)

    def hand_drop(self):
        print("hand Moving to drop position...")
        self.robot_hand.robot_moveL(self.hand_drop_pos, self.speed)
    

    def transform_marker_points(self, maker_point):
        transformed_points = self.cam.transform_marker_points(maker_point)
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
        # keep only real-box IDs
        candidates = [m for m in transformed_points if m["id"] < 100]
        if not candidates:
            print("No box markers found.")
            return None

        top = max(candidates, key=lambda m: m["point"].y)
        print(f"Marker with highest z: ID {top['id']} (z={top['point'].y})")
        return top

    def box_distances(self, markers):
        real_boxes = [m for m in markers if m["id"] < 100]

        for i in range(len(real_boxes)):
            for j in range(i + 1, len(real_boxes)):
                id1 = real_boxes[i]["id"]
                id2 = real_boxes[j]["id"]
                p1 = real_boxes[i]["point"]
                p2 = real_boxes[j]["point"]
                dx = p1.x - p2.x
                dz = p1.z - p2.z
                distance = (dx**2 + dz**2)**0.5
                print(f"ID {id1}-{id2}: x{dx:+.3f} z{dz:+.3f}")
    
    def getrobotDH(self):
        self.robotDH = UR5eDH()
        tool_offset = SE3(0, 0, 0.200)
        self.robotDH.tool = tool_offset

    def gripper_pos_handover(self, marker):
        p = marker["point"]
        approach = [p.x, p.y - 0.20, p.z] + RPY
        pick_pos = [p.x, p.y, p.z] + GRAP_RPY

        print(f"[PICK] Marker ID {marker['id']} at (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)
        tcp_pose_goal = self.get_robot_TCP('gripper')
        pos_current_goal = tcp_pose_goal[:3] + GRAP_RPY
        self.robot_gripper.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
        self.robot_gripper.my_robot_moveL(self.robotDH, pick_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.close_gripper()
        time.sleep(3)
        # retract back to approach pose
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)

    def pick_box(self, marker):
        p = marker["point"]
        approach = [p.x, p.y - 0.20, p.z] + RPY
        pick_pos = [p.x, p.y, p.z] + GRAP_RPY

        print(f"[PICK] Marker ID {marker['id']} at (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)
        tcp_pose_goal = self.get_robot_TCP('gripper')
        pos_current_goal = tcp_pose_goal[:3] + GRAP_RPY
        self.robot_gripper.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
        self.robot_gripper.my_robot_moveL(self.robotDH, pick_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.close_gripper()
        time.sleep(3)
        # retract back to approach pose
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)


    def place_box(self, marker_point):
        p = marker_point
        # approach pose (X–Z move, Y fixed)
        approach = [p.x, p.y - 0.20, p.z] + RPY
        # actual place pose
        place_pos = [p.x, p.y - 0.073, p.z] + GRAP_RPY
        print(f"[PLACE] at marker pos (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        print("move linear")

        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, True)
        self.robot_gripper.my_robot_moveL(self.robotDH, place_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.open_gripper()
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)

    def place_box_on_hand(self, marker_point):
        p = marker_point
        # approach pose (X–Z move, Y fixed)
        approach = [p.x, p.y - 0.20, p.z] + RPY
        # actual place pose
        place_pos = [p.x+0.065, p.y - 0.073, p.z+0.06] + GRAP_RPY
        print(f"[PLACE] at marker pos (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        print("move linear")

        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        self.robot_gripper.my_robot_moveL(self.robotDH, place_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.open_gripper()
        self.robot_gripper.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)

    def find_next_stack_position(self, current_id, sorted_markers):
        """Finds the next marker that has a higher ID."""
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None


    def sort_pick_and_place(self):

        marker_points = self.cam_relasense()
        transformed_points = self.transform_marker_points(marker_points)
        sorted_markers = sorted(transformed_points, key=lambda m: m["id"])

        for marker in sorted_markers:
            current_id = marker["id"]
            self.pick_box(marker)
            next_marker = self.find_next_stack_position(current_id, sorted_markers)
            if next_marker:
                self.place_box(next_marker["point"])
            else:
                print(f"[SKIP STACK] No higher marker found for marker ID {current_id}")

    
    def stack_chain_boxes(self, target_id=104, count=3):

        # Gather the first N box markers by ascending ID
        raw_pts_all  = self.cam_relasense()
        transformed_all = self.transform_marker_points(raw_pts_all)
        box_markers = sorted([m for m in transformed_all if m['id'] < 100],
                            key=lambda m: m['id'])[:count]

        prev_id = target_id
        for marker in box_markers:
            # Refresh marker map right before placing
            raw_pts     = self.cam_relasense()
            transformed = self.transform_marker_points(raw_pts)
            point_map   = {m['id']: m['point'] for m in transformed}

            if prev_id not in point_map:
                print(f"[STACK] Cannot find updated position for marker {prev_id}, abort chain.")
                return

            place_pt = point_map[prev_id]
            print(f"[STACK] Picking box {marker['id']} → placing at marker {prev_id} position")

            # pick the box
            self.pick_box(marker)
            # place it at the freshly read position
            self.place_box(place_pt)
            # go home
            self.move_home_gripper()
            time.sleep(1)

            # next time, we’ll stack onto the just-picked box’s ID
            prev_id = marker['id']

    def detect_overlaps(self, pts):

        STACK_Y_THRESHOLD = 0.29

        # 1) Filter down to just the real boxes
        real_boxes = [m for m in pts if m["id"] < 100]

        # 2) Collect those whose Y exceeds the threshold
        stacked = [m for m in real_boxes if m["point"].y < STACK_Y_THRESHOLD]

        # 3) Return as one group if non‑empty, else no groups
        return [stacked] if stacked else []
    
    def destack_within_zone(self, group_ids):

        while True:
            raw_pts = self.cam_relasense()
            transformed = self.transform_marker_points(raw_pts)
            self.box_distances(transformed)

            overlaps = self.detect_overlaps(transformed)
            if not overlaps:
                print("[DESTACK] No more stacks in this zone.")
                break

            lowest = self.min_marker(transformed)
            if lowest is None:
                print("[DESTACK] No boxes found.")
                break

            candidates = [
                m for m in transformed
                if m["id"] in group_ids
                and m["id"] not in self.used_empty_markers
            ]
            if not candidates:
                print("[DESTACK] No empty spots in specified group.")
                break

            spot = candidates[0]
            self.used_empty_markers.append(spot["id"])
            print(f"[DESTACK] Moving box {lowest['id']} → marker {spot['id']}")
            self.pick_box(lowest)
            self.robot_gripper_move_home_rpy()
            # refresh spot pose from camera before placing
            raw2 = self.cam_relasense()
            xf2 = self.transform_marker_points(raw2)
            updated = next((m for m in xf2 if m["id"] == spot["id"]), None)
            if updated:
                self.place_box(updated["point"])
            else:
                print(f"[DESTACK] Warning: marker {spot['id']} lost, placing at last known point")
                self.place_box(spot["point"])
            self.move_home_gripper()
            time.sleep(1)

    def pos_for_handover(self):

        # Step 1: Move to gripper handover position with the gripper robot
        print("Moving to GRIPPER_HAND_OVER_POS...")
        self.robot_gripper_position_for_hand_over()
        self.robot_hand_over()
        
        time.sleep(1)

        
        
def main():
    mover = TaskPlanning()
    mover.move_home_gripper()
    mover.move_home_hand()
    time.sleep(2)
    while True:
        # At the start of each loop, check for overlaps
        raw_pts = mover.cam_relasense()
        transformed = mover.transform_marker_points(raw_pts)
        overlaps = mover.detect_overlaps(transformed)
        # Print the detected overlaps for debugging
        if not overlaps:
            # No stacks detected: skip destacking menu and go to arrange phase
            print("[INFO] No stacked boxes detected. Proceeding to Arrange phase.")
            # Jump to Phase 2: Arrange remaining boxes
            break
        # ----- Phase 1: Destack only within the left or right zone -----
        print("Choose an operation:")
        print("  1) Unstack on LEFT zone (100–103)")
        print("  2) Unstack on RIGHT zone (104–107)")
        print("  3) Custom")

        op_choice = input("Enter 1/2/3: ").strip()

        if op_choice == "1":
            mover.destack_within_zone(mover.group_left)
            again = input("Do you want to perform another operation? (y/n): ").strip().lower()
            if again != "y":
                break

        elif op_choice == "2":
            mover.destack_within_zone(mover.group_right)
            again = input("Do you want to perform another operation? (y/n): ").strip().lower()
            if again != "y":
                break

        elif op_choice == "3":
            # User chose custom stacking: skip destack and proceed to Phase 2
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

    # ----- Phase 2: Arrange remaining boxes: CLI menu -----
    while True:
        raw_pts     = mover.cam_relasense()
        transformed = mover.transform_marker_points(raw_pts)

        box_ids     = sorted(m["id"] for m in transformed if m["id"] < 100)
        empty_ids   = sorted(m["id"] for m in transformed if m["id"] >= 100)

        print("\n[ARRANGE] No stacked boxes detected, entering Arrange phase.")
        print("How would you like to stack the remaining boxes?")
        print("  1) min→max IDs:", box_ids)
        print("  2) max→min IDs:", list(reversed(box_ids)))
        print("  3) custom sequence")
        choice = input("Enter 1/2/3: ").strip()

        if choice == "1":
            seq  = box_ids
            dest = int(input(f"Place onto which empty spot? {empty_ids}: ").strip())
        elif choice == "2":
            seq  = list(reversed(box_ids))
            dest = int(input(f"Place onto which empty spot? {empty_ids}: ").strip())
        else:
            # FULLY CUSTOM: last ID is destination, everything before is pick‐order
            print("Boxes you can move:", box_ids)
            print("Possible destinations (boxes or empty):", box_ids + empty_ids)
            tokens = input(
                "Enter IDs where ALL BUT THE LAST are boxes to pick, "
                "and LAST is the destination marker (e.g. '3 7 8 2'): "
            ).split()
            ids   = [int(t) for t in tokens]
            seq   = ids[:-1]
            dest  = ids[-1]

        print(f"\n[ARRANGE] Stacking {seq} → marker {dest}")
        # If user selected handover destination (ID 200), perform handover and exit
        if dest == 200:
            print("[HANDOVER] Moving to handover with box sequence:", seq)
            for bid in seq:
                # find the box point safely
                marker_entry = next((m for m in transformed if m["id"] == bid), None)
                if marker_entry is None:
                    print(f"[ERROR] Box ID {bid} not detected; skipping this handover.")
                    continue
                pt = marker_entry["point"]
                print(f"[HANDOVER] Picking box {bid}")
                mover.pick_box({"id": bid, "point": pt})
                mover.move_home_gripper()

                # move gripper into handover position
                mover.robot_gripper_position_for_hand_over()
                time.sleep(1)

                # move hand robot to initial handover pose
                mover.robot_hand_over()
                time.sleep(1)

                # refresh and detect marker ID 200 on the hand
                raw_hand = mover.cam_relasense()
                trans_hand = mover.transform_marker_points(raw_hand)
                hand_marker = next((m for m in trans_hand if m["id"] == 200), None)
                if hand_marker:
                    print("[HANDOVER] Moving hand robot to updated marker 200 position")
                    # move hand robot to the updated marker 200 position
                else:
                    print("Warning: handover marker ID 200 not found on refresh")

                # release the box onto the hand at marker 200
                if hand_marker:
                    mover.place_box_on_hand(hand_marker["point"])
                else:
                    print("Error: cannot place on hand because marker 200 was not detected")
                time.sleep(1)

                
                mover.move_home_hand()
                mover.hand_drop()
                mover.move_home_hand()
                mover.move_home_gripper()
                time.sleep(1)
            mover.stop_all()
            print("Program completed via handover.")
            return
        for bid in seq:
            # find its point
            pt = next(m["point"] for m in transformed if m["id"] == bid)
            print(f"[STACK] Picking box {bid} → placing at marker {dest}")
            mover.pick_box({"id": bid, "point": pt})

            # optional: re-home with correct RPY
            mover.robot_gripper_move_home_rpy()

            # Single capture for destination marker
            raw_pts2 = mover.cam_relasense()
            trans_pts2 = mover.transform_marker_points(raw_pts2)
            dest_marker = next((m for m in trans_pts2 if m["id"] == dest), None)
            if not dest_marker:
                print(f"[ERROR] Destination marker ID {dest} not found; skipping this placement.")
                continue
            dest_pt = dest_marker["point"]

            mover.place_box(dest_pt)
            mover.move_home_gripper()
            time.sleep(1)

            # next placement uses this box as the new “destination”
            dest = bid

        # end of arrange
        mover.stop_all()
        print("Program completed. Robot and camera released.")
        again = input("again? (y/n): ").strip().lower()
        if again == 'y':
            # Repeat the Arrange loop
            continue
        else:
            break


if __name__ == "__main__":
    main()
