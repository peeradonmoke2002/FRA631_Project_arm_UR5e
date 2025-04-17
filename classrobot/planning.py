import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
from spatialmath import SE3
import os
import csv
import numpy as np
from math import cos, sin

class Planning:

    def __init__(self, dt=0.01):
        self.dt = dt

    def dh(self, alpha, a, d, theta):
        T = np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                    [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                    [0, sin(alpha), cos(alpha), d],
                    [0, 0, 0, 1]])
        return T
    
    def fk_ur(self, q: np.ndarray, dh_para: np.ndarray) -> np.ndarray:
        """
        Forward Kinematics for UR type robots
        :param: dh_para: DH-Transformation, table of dh parameters (alpha, a, d, theta)
        :param: q: Gelenkwinkel
        """
        T_0_6 = np.zeros((4, 4))

        dh_params_count = dh_para.shape[1]
        number_dh_trafos = dh_para.shape[0]

        if dh_params_count != 4:
            print("Wrong number of dh parameters!")
            return None

        trafo_matrizes = []

        for i in range(number_dh_trafos):
            trafo_matrizes.append(self.dh(dh_para[i, 0], dh_para[i, 1], dh_para[i, 2], q[i]))

        if len(trafo_matrizes) != 0:
            for i in range(len(trafo_matrizes) - 1):
                if i == 0:
                    T_0_6 = trafo_matrizes[i] @ trafo_matrizes[i+1]
                else:
                    T_0_6 = T_0_6 @ trafo_matrizes [i+1]

        return T_0_6
    
    def jacobian_matrix(self,q: np.ndarray, dh_para: np.ndarray, runden = False) -> np.array:       # code in the video decription

        Jacobian = np.zeros((6, 6))

        T_0_6 = self.fk_ur(q, dh_para)               # transformation matrix of the system (forward kinematics)
        point_end = T_0_6[0:3, 3]               # calculate the TCP origin coordinates

        T_0_i = np.array([[1, 0, 0, 0],         # create T_0_0; needed for for-loop
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        for i in range(6):

            if i == 0:                          # kinematic chain
                T_0_i = T_0_i                   # adds velocity of previous joint to current joint
            else:                               # using the DH parameters
                T = self.dh(dh_para[i-1, 0], dh_para[i-1, 1], dh_para[i-1, 2], q[i-1])
                T_0_i = np.dot(T_0_i, T)

            z_i = T_0_i[0:3, 2]                 # gets the vectors p_i and z_i for the Jacobian from the last two coloums of the transformation matrices  
            p_i = T_0_i[0:3, 3]
            r = point_end - p_i
            Jacobian[0:3, i] = np.cross(z_i, r) # linear portion
            Jacobian[3:6, i] = z_i              # angular portion             ## each time the loop is passed, another column of the Jacobi matrix is filled

            if runden:
                Jacobian[0:3, i] = np.round(np.cross(z_i, r), 3)              # round if True
                Jacobian[3:6, i] = np.round(z_i, 3)

        return Jacobian
    
    def cubic_trajectory(self, p0, p1, v0, v1, T):
        p0, p1, v0, v1 = map(np.array, (p0, p1, v0, v1))
        num_steps = int(np.ceil(T / self.dt)) + 1
        t = np.linspace(0, T, num_steps).reshape(-1, 1)
        a = (2 * (p0 - p1) + (v0 + v1) * T) / (T ** 3)
        b = (3 * (p1 - p0) - (2 * v0 + v1) * T) / (T ** 2)
        pos = a * t**3 + b * t**2 + v0 * t + p0
        vel = 3 * a * t**2 + 2 * b * t + v0
        acc = 6 * a * t + 2 * b
        return t.flatten(), pos, vel, acc

    def compute_traj_time(self, d, v_tool, a_max):
        if d < (v_tool ** 2) / a_max:
            T_total = 2 * np.sqrt(d / a_max)
            profile = 'Triangular'
        else:
            T_accel = v_tool / a_max
            d_accel = 0.5 * a_max * T_accel ** 2
            d_ramps = 2 * d_accel
            T_total = 2 * T_accel + (d - d_ramps) / v_tool
            profile = 'Trapezoidal'
        return T_total, profile

    def pose_to_matrix(self, pose):
        tx, ty, tz, roll, pitch, yaw = pose
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def plot_waypoints(self, waypoints, pos_start, pos_end):
        pos = np.array([T.t for T in waypoints])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='red', label="Waypoints", s=1)
        ax.scatter(pos_start[0], pos_start[1], pos_start[2], color='green', s=50, label="Start")
        ax.scatter(pos_end[0], pos_end[1], pos_end[2], color='blue', s=50, label="End")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        plt.show()

    def plot_speeds(self, t_vec, speed_traj, traj_T):
        angular_speeds = np.zeros((len(traj_T), 3))
        for i in range(1, len(traj_T)):
            delta_T = traj_T[i-1].inv() * traj_T[i]
            angular_speeds[i, :] = delta_T.angvec()[0] / self.dt
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t_vec, speed_traj[:, 0], 'r-', label='Linear X')
        plt.plot(t_vec, speed_traj[:, 1], 'g-', label='Linear Y')
        plt.plot(t_vec, speed_traj[:, 2], 'b-', label='Linear Z')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(t_vec, angular_speeds[:, 0], 'r-', label='Angular X')
        plt.plot(t_vec, angular_speeds[:, 1], 'g-', label='Angular Y')
        plt.plot(t_vec, angular_speeds[:, 2], 'b-', label='Angular Z')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def log_waypoints(self, waypoints, path):
        file_path = pathlib.Path(path) / "waypoints_log.csv"
        if os.path.exists(file_path):
            raise FileExistsError(f"{file_path} already exists.")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z", "Roll", "Pitch", "Yaw"])
            for wp in waypoints:
                writer.writerow(wp.t.tolist() + list(wp.rpy()))
        print(f"Waypoints logged to {file_path}")

    def log_joint_trajectory(self, joint_trajectory, path):
        file_path = pathlib.Path(path) / "joint_trajectory_log.csv"
        if os.path.exists(file_path):
            raise FileExistsError(f"{file_path} already exists.")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"])
            for q in joint_trajectory:
                writer.writerow(q)
        print(f"Joint trajectory logged to {file_path}")

    def load_waypoints(self, path):
        file_path = pathlib.Path(path) / "waypoints_log.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        waypoints = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                x, y, z, roll, pitch, yaw = map(float, row)
                T = SE3(x, y, z) * SE3.RPY(roll, pitch, yaw)
                waypoints.append(T)
        print(f"Waypoints loaded from {file_path}")
        return waypoints
    
    def compute_joint_trajectory(self, traj_T, real_robot):
        joint_trajectory = []
        for idx, T_pose in enumerate(traj_T):
            tcp_pose = T_pose.t.tolist() + list(T_pose.rpy())
            q = real_robot.robot_get_ik(tcp_pose)
            if q is None:
                raise RuntimeError(f"IK failed for waypoint {idx}")
            joint_trajectory.append(q)
        print("Joint trajectory computed successfully.")
        return joint_trajectory

    def compute_joint_trajectory_from_q_robot(self, q_robot):
        joint_trajectory = []
        for idx, T_pose in enumerate(q_robot):
            q = q_robot[idx]
            if q is None:
                raise RuntimeError(f"IK failed for waypoint {idx}")
            joint_trajectory.append(q)
        print("Joint trajectory computed successfully.")
        return joint_trajectory
    
    def load_joint_trajectory(self, path):
        file_path = pathlib.Path(path) / "joint_trajectory_log.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        joint_trajectory = []
        df = pd.read_csv(file_path)
        # Extract joint configurations from the DataFrame
        for index, row in df.iterrows():
            joint_config = row.values.tolist()  # Convert the row to a list
            joint_trajectory.append(joint_config)
        return joint_trajectory
