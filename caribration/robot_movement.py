import rtde_control
import rtde_receive
import time
import math

class UR5eRobotCARIBRATION:
    def __init__(self, ip: str):
        """
        Connect to the UR5e robot using its IP address.
        """
        self.ip = ip
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        print(f"Connected to UR5e at {ip}")

    def moveL(self, pose: list, speed: float = 0.25, acceleration: float = 1.2):
        """
        Command the robot to move linearly (moveL) to the given pose.
        pose: list of 6 values [x, y, z, rx, ry, rz].
        speed: linear speed in m/s.
        acceleration: linear acceleration in m/s².
        """
        print(f"Moving linearly to pose: {pose}")
        self.rtde_c.moveL(pose, speed, acceleration)
        time.sleep(1)  # wait a bit for the motion to settle

    def moveL_until_limit(self, start_pose: list, direction: list,
                          distance_limit: float, step_distance: float,
                          capture_status_func=None, state_delay: float = 0.0,
                          speed: float = 0.25, acceleration: float = 1.2) -> list:
        """
        Move linearly from start_pose in the given direction until the total distance_limit is reached.
        Each movement is in increments of step_distance.
        
        After each move, wait for a capture status and an additional delay (state_delay) before moving to the next pose.
        
        Parameters:
            start_pose (list): Starting pose [x, y, z, rx, ry, rz].
            direction (list): Direction vector for movement [dx, dy, dz] (only position is updated).
            distance_limit (float): Total distance to move (in meters).
            step_distance (float): Distance between successive moveL commands (in meters).
            capture_status_func (callable, optional): A function that returns True when capture is complete.
                                                      If None, no capture wait is performed.
            state_delay (float): Additional delay (in seconds) after capture status is confirmed before moving on.
            speed (float): Linear speed (in m/s).
            acceleration (float): Linear acceleration (in m/s²).
        
        Returns:
            List of commanded poses.
        """
        commanded_poses = []
        current_pose = start_pose.copy()
        total_distance = 0.0

        # Normalize the direction vector.
        mag = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        if mag == 0:
            raise ValueError("Direction vector cannot be zero.")
        norm_direction = [d / mag for d in direction]

        while total_distance < distance_limit:
            # Compute the next pose (update only x, y, z; orientation remains unchanged).
            next_pose = current_pose.copy()
            next_pose[0] += norm_direction[0] * step_distance
            next_pose[1] += norm_direction[1] * step_distance
            next_pose[2] += norm_direction[2] * step_distance

            self.moveL(next_pose, speed, acceleration)
            commanded_poses.append(next_pose)
            
            # Wait until capture status is received if a function is provided.
            if capture_status_func is not None:
                print("Waiting for capture status...")
                while not capture_status_func():
                    time.sleep(0.1)
                print("Capture complete.")

            # Additional delay per state.
            if state_delay > 0:
                print(f"Waiting additional state delay of {state_delay} seconds...")
                time.sleep(state_delay)
            
            current_pose = next_pose
            total_distance += step_distance

        return commanded_poses

    def get_current_pose(self) -> list:
        """
        Return the robot's current TCP pose as a list [x, y, z, rx, ry, rz].
        """
        return self.rtde_r.getActualTCPPose()

    def disconnect(self):
        """
        Disconnect the RTDE control interface.
        """
        self.rtde_c.disconnect()
        print("Disconnected from UR5e")
