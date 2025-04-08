import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from .point3d import Point3D

# Load advanced configuration for RealSense from a JSON file.
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cam.json")
jsonObj = json.load(open(config_path))
json_string = str(jsonObj).replace("'", '\"')

class RealsenseCam:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Enable IMU streams if needed.
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        dev = self.profile.get_device()
        advnc_mode = rs.rs400_advanced_mode(dev)
        if not advnc_mode.is_enabled():
            advnc_mode.toggle_advanced_mode(True)
        advnc_mode.load_json(json_string)
        print("RealSense camera started with aligned color, depth and IMU streams.")

    def get_color_and_depth_frames(self) -> tuple:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        # Get intrinsics from the color stream.
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame, intrinsics

    def get_imu_orientation(self):
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            motion_data = accel_frame.as_motion_frame().get_motion_data()
            measured = np.array([motion_data.x, motion_data.y, motion_data.z])
            norm = np.linalg.norm(measured)
            if norm == 0:
                return np.eye(3)
            measured_norm = measured / norm
            desired = np.array([0, 0, 1])
            R_corr, _ = R.align_vectors([desired], [measured_norm])
            return R_corr.as_matrix()
        return np.eye(3)

    def get_board_pose_with_rotation(self, aruco_dict, marker_length=0.04):
        color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
        if color_image is None or depth_frame is None:
            print("Failed to capture color/depth.")
            return None, Point3D(0, 0, 0), None

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            camera_matrix = np.array([
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)

            ret, rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )

            if ret is not None and len(rvecs) > 0:
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Draw axes using the default cv2.drawFrameAxes function.
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                point3d = Point3D(tvec[0], tvec[1], tvec[2])
                return color_image, point3d, rotation_matrix

        print("No markers detected.")
        return color_image, Point3D(0, 0, 0), None

    def stop(self):
        self.pipeline.stop()
        print("RealSense camera stopped.")

    def restart(self):
        self.stop()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("RealSense camera restarted with aligned color, depth and IMU streams.")
