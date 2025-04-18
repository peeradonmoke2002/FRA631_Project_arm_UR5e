import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
from spatialmath import SE3
import os
import csv
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np


class Planning:

    def __init__(self, dt=0.01):
        self.dt = dt
    
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

    def cubic_trajectory_v1(self, p0, p1, v0, v1, T, dt):
        # ensure numpy arrays
        p0, p1, v0, v1 = map(np.asarray, (p0, p1, v0, v1))
        # time vector
        t = np.arange(0, T + 1e-8, dt)        # include T
        ts = t.reshape(-1, 1)                 # for broadcasting
        
        # coefficients
        a0 = p0
        a1 = v0
        a2 =  3*(p1 - p0)/T**2 - (2*v0 + v1)/T
        a3 = -2*(p1 - p0)/T**3 + (v0 + v1)/T**2
        
        # evaluate
        pos = a0 + a1*ts + a2*ts**2 + a3*ts**3
        vel =      a1   + 2*a2*ts   + 3*a3*ts**2
        acc =           2*a2       + 6*a3*ts

        return t.flatten(), pos, vel, acc

    
    def cubic_trajectory_v2(self, p0, p1, v0, v1, q0, q1, T):

        # --- Translation (cubic) ---
        p0, p1, v0, v1 = map(np.array, (p0, p1, v0, v1))
        num_steps = int(np.ceil(T / self.dt)) + 1
        t = np.linspace(0, T, num_steps).reshape(-1, 1)

        a = (2 * (p0 - p1) + (v0 + v1) * T) / (T ** 3)
        b = (3 * (p1 - p0) - (2 * v0 + v1) * T) / (T ** 2)

        pos = a * t**3 + b * t**2 + v0 * t + p0
        vel = 3 * a * t**2 + 2 * b * t + v0
        acc = 6 * a * t + 2 * b

        # --- Orientation (SLERP) ---
        # Normalize quaternions
        q0 = np.asarray(q0) / np.linalg.norm(q0)
        q1 = np.asarray(q1) / np.linalg.norm(q1)

        # Build SLERP object
        key_rots = R.from_quat([q0, q1])
        slerp = Slerp([0.0, 1.0], key_rots)

        # Normalize time to [0,1]
        tau = t.flatten() / T

        # Interpolate orientations
        orients = slerp(tau).as_quat()

        return t.flatten(), pos, vel, acc, orients
    

    def cubic_trajectory_v3(self, p0, p1, v0, v1, rpy0, rpy1, T):
        """
        Generate a cubic (3rd-order) trajectory for translation and SLERP for orientation,
        using RPY inputs and returning RPY outputs.

        Parameters:
        -----------
        p0   : array_like, shape (3,)
               Start position [x, y, z]
        p1   : array_like, shape (3,)
               End position [x, y, z]
        v0   : array_like, shape (3,)
               Start velocity [vx, vy, vz]
        v1   : array_like, shape (3,)
               End velocity [vx, vy, vz]
        rpy0 : array_like, shape (3,)
               Start orientation as roll, pitch, yaw (rad)
        rpy1 : array_like, shape (3,)
               End orientation as roll, pitch, yaw (rad)
        T    : float
               Total trajectory time

        Returns:
        --------
        t        : ndarray, shape (N,)
                   Time vector
        pos      : ndarray, shape (N,3)
                   Positions along the cubic trajectory
        vel      : ndarray, shape (N,3)
                   Velocities along the cubic trajectory
        acc      : ndarray, shape (N,3)
                   Accelerations along the cubic trajectory
        rpy_traj : ndarray, shape (N,3)
                   SLERP-interpolated roll, pitch, yaw angles (rad)
        """
        # Translation (cubic)
        p0, p1, v0, v1 = map(np.array, (p0, p1, v0, v1))
        num_steps = int(np.ceil(T / self.dt)) + 1
        t = np.linspace(0, T, num_steps).reshape(-1, 1)

        a = (2 * (p0 - p1) + (v0 + v1) * T) / (T ** 3)
        b = (3 * (p1 - p0) - (2 * v0 + v1) * T) / (T ** 2)

        pos = a * t**3 + b * t**2 + v0 * t + p0
        vel = 3 * a * t**2 + 2 * b * t + v0
        acc = 6 * a * t + 2 * b

        # Orientation (SLERP) with RPY I/O
        # Convert RPY to quaternions
        q0 = R.from_euler('xyz', rpy0, degrees=False).as_quat()
        q1 = R.from_euler('xyz', rpy1, degrees=False).as_quat()

        # Build SLERP
        key_rots = R.from_quat([q0, q1])
        slerp = Slerp([0.0, 1.0], key_rots)
        tau = t.flatten() / T

        # Interpolate and convert back to RPY
        quats = slerp(tau).as_quat()
        rpy_traj = R.from_quat(quats).as_euler('xyz', degrees=False)

        return t.flatten(), pos, vel, acc, rpy_traj
    
    def quintic_trajectory(self, p0, p1, v0, v1, q0, q1, T):
        """
        Generate a quintic (5th-order) SE(3) trajectory from p0 to p1 and q0 to q1
        with boundary conditions on position, velocity, and zero acceleration.
        
        Returns:
        t       – (N,) time vector
        pos     – (N, len(p0)) positions
        vel     – (N, len(p0)) velocities
        acc     – (N, len(p0)) accelerations
        orients – (N,4) quaternions (x,y,z,w)
        """
        # Ensure inputs are numpy arrays
        p0, p1, v0, v1 = map(np.array, (p0, p1, v0, v1))
        q0 = np.asarray(q0) / np.linalg.norm(q0)
        q1 = np.asarray(q1) / np.linalg.norm(q1)

        # Time vector
        num_steps = int(np.ceil(T / self.dt)) + 1
        t = np.linspace(0, T, num_steps)
        ts = t.reshape(-1, 1)

        # Quintic translation coefficients (zero accel endpoints)
        a0 = p0
        a1 = v0
        a2 = np.zeros_like(p0)

        # Powers of T
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5

        # Solve for a3, a4, a5
        M = np.array([
            [   T3,    T4,    T5],
            [ 3*T2,  4*T3,  5*T4],
            [ 6*T , 12*T2, 20*T3]
        ])
        M_inv = np.linalg.inv(M)
        dp = p1 - p0
        rhs = np.vstack([
            dp - v0 * T,
            v1 - v0,
            np.zeros_like(p0)
        ])
        a3, a4, a5 = M_inv.dot(rhs)

        # Evaluate position, velocity, acceleration
        pos = a0 + a1*ts + a2*ts**2 + a3*ts**3 + a4*ts**4 + a5*ts**5
        vel =    a1 + 2*a2*ts   + 3*a3*ts**2 + 4*a4*ts**3 + 5*a5*ts**4
        acc =       2*a2       + 6*a3*ts     +12*a4*ts**2  +20*a5*ts**3

        return t, pos, vel, acc
    

    def quintic_trajectory_v2(self, p0, p1, v0, v1, rpy0, rpy1, T):
        """
        Generate a quintic (5th-order) SE(3) trajectory from p0 to p1
        and from rpy0 to rpy1, with boundary conditions on position,
        velocity, and zero acceleration.

        Parameters:
        -----------
        p0   : array_like, shape (3,)
               Start position [x, y, z]
        p1   : array_like, shape (3,)
               End position [x, y, z]
        v0   : array_like, shape (3,)
               Start velocity [vx, vy, vz]
        v1   : array_like, shape (3,)
               End velocity [vx, vy, vz]
        rpy0 : array_like, shape (3,)
               Start orientation as roll, pitch, yaw (rad)
        rpy1 : array_like, shape (3,)
               End orientation as roll, pitch, yaw (rad)
        T    : float
               Total trajectory duration (s)

        Returns:
        --------
        t       : ndarray, shape (N,)
                  Time vector
        pos     : ndarray, shape (N,3)
                  Positions along the quintic trajectory
        vel     : ndarray, shape (N,3)
                  Velocities along the quintic trajectory
        acc     : ndarray, shape (N,3)
                  Accelerations along the quintic trajectory
        rpy_traj: ndarray, shape (N,3)
                  Interpolated roll, pitch, yaw angles (rad)
        """
        # --- ensure array inputs ---
        p0, p1, v0, v1 = map(np.array, (p0, p1, v0, v1))
        rpy0 = np.array(rpy0)
        rpy1 = np.array(rpy1)

        # --- time vector ---
        num_steps = int(np.ceil(T / self.dt)) + 1
        t = np.linspace(0, T, num_steps)
        ts = t.reshape(-1, 1)

        # --- quintic translation ---
        a0 = p0
        a1 = v0
        a2 = np.zeros_like(p0)

        # Powers of T
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5

        # Reduced system for a3, a4, a5
        M = np.array([
            [   T3,    T4,    T5],
            [ 3*T2,  4*T3,  5*T4],
            [ 6*T , 12*T2, 20*T3]
        ])
        rhs = np.vstack([
            (p1 - p0) - v0 * T,
            v1 - v0,
            np.zeros_like(p0)
        ])
        a3, a4, a5 = np.linalg.inv(M).dot(rhs)

        pos = a0 + a1*ts + a2*ts**2 + a3*ts**3 + a4*ts**4 + a5*ts**5
        vel =    a1 + 2*a2*ts   + 3*a3*ts**2 + 4*a4*ts**3 + 5*a5*ts**4
        acc =       2*a2       + 6*a3*ts     +12*a4*ts**2  +20*a5*ts**3

        # --- SLERP orientation from RPY ---
        # Convert RPY to quaternions
        q0 = R.from_euler('xyz', rpy0, degrees=False).as_quat()
        q1 = R.from_euler('xyz', rpy1, degrees=False).as_quat()

        # SLERP setup
        slerp = Slerp([0.0, 1.0], R.from_quat([q0, q1]))
        tau = t / T
        quats = slerp(tau).as_quat()  # array (N,4)

        # Convert back to RPY
        rpy_traj = R.from_quat(quats).as_euler('xyz', degrees=False)

        return t, pos, vel, acc, rpy_traj

    def slerp_orientation(t: np.ndarray, q0: np.array, q1: np.array, T: float) -> np.ndarray:
        """
        SLERP-interpolate between two unit quaternions over timestamps t.

        Parameters:
        -----------
        t  : array_like, shape (N,)
            Time samples (0 <= t_i <= T).
        q0 : array_like, shape (4,)
            Start quaternion (x, y, z, w).
        q1 : array_like, shape (4,)
            End quaternion (x, y, z, w).
        T  : float
            Total interpolation time.

        Returns:
        --------
        orients : ndarray, shape (N,4)
            Interpolated quaternions at each t[i].
        """
        t = np.asarray(t)
        if np.any(t < 0) or np.any(t > T):
            raise ValueError("All t values must lie within [0, T].")

        # Normalize input quaternions
        q0 = np.asarray(q0) / np.linalg.norm(q0)
        q1 = np.asarray(q1) / np.linalg.norm(q1)

        # Build SLERP object over [0,1]
        key_rots = R.from_quat([q0, q1])
        slerp = Slerp([0.0, 1.0], key_rots)

        # Normalize t to [0,1]
        tau = t / T

        # Compute SLERP and return as (x,y,z,w)
        return slerp(tau).as_quat()

    # def compute_traj_time(self, d, v_tool, a_max):
    #     if d < (v_tool ** 2) / a_max:
    #         T_total = 2 * np.sqrt(d / a_max)
    #         profile = 'Triangular'
    #     else:
    #         T_accel = v_tool / a_max
    #         d_accel = 0.5 * a_max * T_accel ** 2
    #         d_ramps = 2 * d_accel
    #         T_total = 2 * T_accel + (d - d_ramps) / v_tool
    #         profile = 'Trapezoidal'
    #     return T_total, profile
    

    def compute_traj_time(self, d, v_tool, a_max):

        # --- Sanity checks ---
        if d < 0:
            raise ValueError("Distance d must be non‑negative")
        if v_tool <= 0 or a_max <= 0:
            raise ValueError("v_tool and a_max must both be > 0")

        # distance needed to accelerate from 0→v_tool and then decel back to 0
        d_ramps = v_tool**2 / a_max

        if d <= d_ramps:
            # never reaches v_tool → triangular
            profile = 'Triangular'
            t_accel = np.sqrt(d / a_max)
            t_const = 0.0
            t_decel = t_accel
        else:
            # trapezoidal: accel → cruise → decel
            profile = 'Trapezoidal'
            t_accel = v_tool / a_max
            t_const = (d - d_ramps) / v_tool
            t_decel = t_accel

        T_total = t_accel + t_const + t_decel
        return T_total, profile

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