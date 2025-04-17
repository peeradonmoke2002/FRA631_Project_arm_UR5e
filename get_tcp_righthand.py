import rtde_receive

def get_tcp_righthand(robot_ip="192.168.200.20"):
    try:
        # Connect to the robot
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Get the TCP pose
        tcp_pose = rtde_r.getActualTCPPose()
        
        return tcp_pose
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    robot_ip = "192.168.200.20"  # Replace with your robot's IP address
    tcp_pose = get_tcp_righthand(robot_ip)
    if tcp_pose:
        print("TCP Pose:", tcp_pose)