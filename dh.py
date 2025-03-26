import time
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from roboticstoolbox import models

# RTDE connection parameters
ROBOT_IP = "192.168.200.10"  # Replace with your UR5's IP address
PORT = 30004                # Default RTDE port for UR robots

# Load the RTDE configuration (ensure your XML outputs joint angles)
conf = rtde_config.ConfigFile("./RTDE_Python_Client_Library/examples/record_configuration.xml")
output_names, output_types = conf.get_recipe("out")

# Establish RTDE connection to the UR5
con = rtde.RTDE(ROBOT_IP, PORT)
con.connect()
con.send_output_setup(output_names, output_types)
con.send_start()

# Use the built-in UR5 model from the Robotics Toolbox for Python
ur5 = models.DH.UR5()

print("Connected to UR5. Retrieving joint angles and computing forward kinematics...\n")

state = con.receive()
if state:
    # Get the joint angles (assumed to be in radians)
    q = state.actual_q
    print("Joint angles:", q)
    
    # Compute forward kinematics using the built-in model
    T = ur5.fkine(q)
    print("Transformation Matrix (Base to Tool):\n", T)
    print("-" * 50)
    
    # Small delay to avoid flooding the console
    time.sleep(0.1)
