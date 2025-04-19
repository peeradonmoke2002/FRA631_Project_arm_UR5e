# move2object.py
import time
import cv2 as cv
import sys
import pathlib
from spatialmath import SE3

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot import gripper
from classrobot.UR5e_DH import UR5eDH

HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7318488600590023, 0.686830145115122, -1.731258978679887]

RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
GRAP_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

class Move2Object():
    def __init__(self):
        
        self.robot_ip = "192.168.200.10"
        self.speed = 0.1
        self.acceleration = 1.2
        self.dt = 0.01
        self.robotDH = None

        # Initialize the robot connection once.
        self.getrobotDH()
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
    
        self.robot.robot_init(self.robot_ip)

        self.cam = realsense_cam.RealsenseCam()

        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()


    def init_gripper(self):
       
        # Initialize the gripper connection
        host = "192.168.200.11"  # Replace with your gripper's IP address
        port = 502              # Typically the default Modbus TCP port
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()

        # Delay slightly longer than the TIME_PROTECTION (0.5 s)
        print("Testing gripper ...", end="")
        self.close_gripper()  # Now this should actuate the close command
        self.open_gripper()    # Test open command
        self._GRIPPER_LEFT_.my_release()
     
    def init_gripper_before_grap(self):
       
        # Initialize the gripper connection
        host = "192.168.200.11"  # Replace with your gripper's IP address
        port = 502              # Typically the default Modbus TCP port
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()

    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        """
        Closes the gripper.
        """
        self.init_gripper_before_grap()
        time.sleep(0.6)
        print("Closing gripper...")
        self._GRIPPER_LEFT_.my_hand_close()
        time.sleep(2)
        self._GRIPPER_LEFT_.my_release()

    def open_gripper(self):
        """
        Opens the gripper.
        """
        self.init_gripper_before_grap()
        print("Opening gripper...")
        time.sleep(0.6)
        self._GRIPPER_LEFT_.my_hand_open()
        time.sleep(2)
        self._GRIPPER_LEFT_.my_release()

    def cam_relasense(self):
        aruco_dict_type = cv.aruco.DICT_5X5_1000
        point3d = self.cam.cam_capture_marker(aruco_dict_type)
        return point3d
    
    def get_robot_TCP(self):
        """
        Connects to the robot and retrieves the current TCP (end-effector) position.
        Returns a 3-element list: [x, y, z].
        """
        pos = self.robot.robot_get_position()
        print("Robot TCP position:", pos)
        return pos

    def move_home(self):
        print("Moving to home position...")
        time.sleep(2)
        # self.robot.robot_moveL(HOME_POS, self.speed)
        self.robot.my_robot_moveL(self.robotDH, HOME_POS, self.dt, self.speed, self.acceleration, False)
  
    def getrobotDH(self):
        self.robotDH = UR5eDH()
        tool_offset = SE3(0, 0, 0.200)
        self.robotDH.tool = tool_offset

    def move_muti_to_object(self):
        """
        Moves the robot to the specified object position in the camera coordinate system.
        """        
        maker_point = self.cam_relasense()
        print(maker_point)
        transfrom_point = self.cam.transform_marker_points(maker_point)
        print(transfrom_point)
        # Sort the markers by their id in ascending order.
        sorted_markers = sorted(transfrom_point, key=lambda m: m["id"])

        for marker in sorted_markers:
            marker_id = marker["id"]
            point = marker["point"]  # This is an instance of Point3D.
            target_pose_up = [point.x + 0.05, point.y-0.20, point.z] + RPY
            target_pose_down = [point.x + 0.05, point.y, point.z] + GRAP_RPY
            print(f"Moving to marker ID {marker_id} at position {target_pose_up}")
            
            # ----- start movement ----
            self.robot.my_robot_moveL(self.robotDH, target_pose_up, self.dt, self.speed, self.acceleration, False)
            time.sleep(3)
            tcp_pose_goal = self.get_robot_TCP()
            pos_current_goal = tcp_pose_goal[:3]+ GRAP_RPY
            self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
            time.sleep(3)
            self.robot.my_robot_moveL(self.robotDH, target_pose_down, self.dt, self.speed, self.acceleration, False)
            # gripper close
            self.close_gripper()
            time.sleep(3)
            self.robot.my_robot_moveL(self.robotDH, target_pose_up, self.dt, self.speed, self.acceleration, False)
            time.sleep(3)
            tcp_pose_goal = self.get_robot_TCP()
            pos_current_goal = tcp_pose_goal[:3]+ RPY
            self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
            time.sleep(3)
            self.move_home()
            time.sleep(3)
            self.robot.my_robot_moveL(self.robotDH, target_pose_up, self.dt, self.speed, self.acceleration, False)
            time.sleep(3)
            tcp_pose_goal = self.get_robot_TCP()
            pos_current_goal = tcp_pose_goal[:3]+ GRAP_RPY
            self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
            time.sleep(3)
            self.robot.my_robot_moveL(self.robotDH, target_pose_down, self.dt, self.speed, self.acceleration, False)
            # gripper open
            self.open_gripper()
            time.sleep(3)
            self.robot.my_robot_moveL(self.robotDH, target_pose_up, self.dt, self.speed, self.acceleration, False)
            time.sleep(3)
            tcp_pose_goal = self.get_robot_TCP()
            pos_current_goal = tcp_pose_goal[:3]+ RPY
            self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
            time.sleep(3)
            self.move_home()
            time.sleep(3)
            # done :)

            # ----- this code is work -----
            # up over the object
            # self.robot.robot_moveL(target_pose_up, self.speed)
            # time.sleep(5)
            # # down to the object
            # self.robot.robot_moveL(target_pose_down, self.speed)
            # # gripper close
            # self.close_gripper()
            # time.sleep(3)
            # # back to home position
            # self.move_home()
            # time.sleep(3)
            # # up over the object
            # self.robot.robot_moveL(target_pose_up, self.speed)
            # time.sleep(3)
            # # down to the object
            # self.robot.robot_moveL(target_pose_down, self.speed)
            # # gripper open
            # self.open_gripper()
            # time.sleep(3)
            # # back to home position
            # self.move_home()
            # time.sleep(3)
            # done :)
            print(f"Completed move to marker ID {marker_id}")

def main():
    try:
        move2object = Move2Object()

        move2object.move_home()
        time.sleep(2)
        move2object.move_muti_to_object()
        time.sleep(5)  # Added delay before returning home
        move2object.move_home()  # Return to home position after moving to object
        move2object.stop_all()  # Stop all connections and clean up

    except Exception as e:
        print(f"Error initializing Move2Object: {e}")
        return


if __name__ == "__main__":
    main()
