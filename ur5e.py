import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import time
import math
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import roboticstoolbox as rt
from roboticstoolbox import models
from spatialmath.base import angvec2tr

ROBOT_IP = "192.168.200.10"  # Replace with your UR5's IP address
PORT = 30004                 # Default RTDE port for UR robots

# Load the RTDE configuration (ensure your XML outputs joint angles & TCP pose)
conf = rtde_config.ConfigFile("./RTDE_Python_Client_Library/examples/record_configuration.xml")
output_names, output_types = conf.get_recipe("out")

# Establish RTDE connection to the UR5
con = rtde.RTDE(ROBOT_IP, PORT)
con.connect()
con.send_output_setup(output_names, output_types)
con.send_start()
state = con.receive()
q = state.actual_q  # Retrieve the current joint angles

# Define the robot links using DH parameters

# Link 1
L1 = RevoluteDH(
    d=0.1625,
    a=0,
    alpha=np.pi/2,
    m=3.761,
    r=[0, -0.02561, 0.00193],
    I=np.zeros((3, 3))
)

# Link 2
L2 = RevoluteDH(
    d=0,
    a=-0.425,
    alpha=0,
    m=8.058,
    r=[0.2125, 0, 0.11336],
    I=np.zeros((3, 3))
)

# Link 3
L3 = RevoluteDH(
    d=0,
    a=-0.3922,
    alpha=0,
    m=2.846,
    r=[0.15, 0, 0.0265],
    I=np.zeros((3, 3))
)

# Link 4
L4 = RevoluteDH(
    d=0.1333,
    a=0,
    alpha=np.pi/2,
    m=1.37,
    r=[0, -0.0018, 0.01634],
    I=np.zeros((3, 3))
)

# Link 5
L5 = RevoluteDH(
    d=0.0997,
    a=0,
    alpha=-np.pi/2,
    m=1.3,
    r=[0, 0.0018, 0.01634],
    I=np.zeros((3, 3))
)

# Link 6
I6 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0.0002]
])
L6 = RevoluteDH(
    d=0.0996,
    a=0,
    alpha=0,
    m=0.365,
    r=[0, 0, -0.001159],
    I=I6
)

# Create the UR5e robot model
UR5e = DHRobot([L1, L2, L3, L4, L5, L6], name='UR5e')
tool_offset = SE3(0, 0, 0.2)
UR5e.tool = tool_offset  # Incorporate this offset into forward kinematics

# Print robot details and the current joint configuration
print(UR5e)
print("Joint angles:", q)

# Compute forward kinematics for the current joint angles
T = np.array(UR5e.fkine(q).A)
print("End-effector pose (with tool offset):")
print(T)

# Define a transformation matrix from the robot frame to the world frame
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0.0575],
    [1, 0, 0, 0.4],
    [0, 0, 0, 1]
])
# # Compute the final transformation (world frame pose)
result = T @ A
print("End-effector pose in world frame:")
print(result)



# # Define a valid 4x4 transformation matrix N
# N = np.array([
#     [1, 0, 0, 1],
#     [0, 1, 0, 0.3],
#     [0, 0, 1, 0.4],
#     [0, 0, 0, 1]
# ])
# # Perform inverse kinematics
# T_inv = UR5e.ikine_LM(N)
# print("Inverse kinematics:")
# print(T_inv)

