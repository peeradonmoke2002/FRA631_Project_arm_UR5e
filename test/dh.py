import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # สำหรับการ plot 3D
import numpy as np
import rtde_receive
# ================= UR5e Forward Kinematics =================
L1 = rtb.RevoluteDH(d=0.1625, a=0,       alpha=np.pi/2)
L2 = rtb.RevoluteDH(d=0,      a=-0.425,  alpha=0)
L3 = rtb.RevoluteDH(d=0,      a=-0.3922, alpha=0)
L4 = rtb.RevoluteDH(d=0.1333, a=0,       alpha=np.pi/2)
L5 = rtb.RevoluteDH(d=0.0997, a=0,       alpha=-np.pi/2)
L6 = rtb.RevoluteDH(d=0.0996, a=0,       alpha=0)

robot = rtb.DHRobot([L1, L2, L3, L4, L5, L6], name='UR5e')


robot_ip = "192.168.200.10"
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, 30004)


# define joint angles from teach pendant degree

# joint_deg = [41.24, -111.09, -117.58, -130.54, -50.14, 135.40]
joint_deg = rtde_r.getActualQ()
q = np.array(rtde_r.getActualQ())


# calculate forward kinematics (flange pose) in robot base frame
T_fk = robot.fkine(q)

# set tool offset: translate 200 mm by z from flange frame
T_tool = SE3(0, 0, 0.200)  # 0.200 m = 200 mm

# calculate TCP pose (flange + tool offset) in robot base frame
T_TCP = T_fk * T_tool

# convert orientation from RPY radians to degrees
rpy_tcp = np.degrees(T_TCP.rpy())

print("=== TCP Pose in Robot Base Frame ===")
print("Transformation Matrix:")
print(T_TCP)
print("\nOrientation (RPY in degrees):")
print(f"Roll:  {rpy_tcp[0]:.3f}°")
print(f"Pitch: {rpy_tcp[1]:.3f}°")
print(f"Yaw:   {rpy_tcp[2]:.3f}°")

# ================= World Frame Transformation =================


T_world = SE3(0.06, 0.4, 0) * SE3.Rx(1.209) * SE3.Ry(-1.209) * SE3.Rz(1.209)

T_inv = T_world.inv()
T_TCP_world = T_TCP

rpy_tcp_world = np.degrees(T_TCP_world.rpy())

print("\n=== TCP Pose in World Frame ===")
print("Transformation Matrix:")
print(T_TCP_world)
print("\nOrientation (RPY in degrees):")
print(f"Roll:  {rpy_tcp_world[0]:.3f}°")
print(f"Pitch: {rpy_tcp_world[1]:.3f}°")
print(f"Yaw:   {rpy_tcp_world[2]:.3f}°")

print(joint_deg )