import pyrealsense2 as rs
import cv2
import numpy as np
import math
from .point3d import Point3D
import json
import pathlib
from scipy.spatial.transform import Rotation as R
import math 
import time
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Dict


# global variables
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs/cam.json"
CONFIG_MATRIX_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs/best_matrix.json"
jsonObj = json.load(open(CONFIG_PATH))
CAM_CONFIG_JSON = str(jsonObj).replace("'", '\"')
print("Loading camera configuration from:", CONFIG_PATH)
print("Loading camera matrix from:", CONFIG_MATRIX_PATH)


class RealsenseCam:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.align_depth = None 
        self.imu_pipe = None
        self.imu_config = None
        self.best_matrix = None
        self.init_cam()
        self.load_matrix()
        print("RealsenseCam initialized with width: {}, height: {}, fps: {}".format(width, height, fps))
  
    def init_cam(self):
        """
        Initialize the RealSense camera pipeline with color, depth, and motion streams.
        This method is called in the constructor.
        """
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.imu_pipe = rs.pipeline()
            self.imu_config = rs.config()

            if self.pipeline is None or self.config is None:
                raise RuntimeError("Failed to create RealSense pipeline or config.")

            # Enable color and depth streams before starting the pipeline.
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Enable motion streams without specifying fps; device defaults will be used.
            self.imu_config.enable_stream(rs.stream.gyro)
            self.imu_config.enable_stream(rs.stream.accel)

            # Start the pipeline and get the profile.
            self.profile = self.pipeline.start(self.config)
            self.imu_pipe.start(self.imu_config)

            # Align depth to color.
            self.align = rs.align(rs.stream.color)
            self.align_depth = rs.align(rs.stream.depth)

            # Get device and enable advanced mode.
            dev = self.profile.get_device()
            if not dev:
                raise RuntimeError("Failed to get RealSense device.")

            advnc_mode = rs.rs400_advanced_mode(dev)
            if not advnc_mode.is_enabled():
                print("Enabling advanced mode...")
                advnc_mode.toggle_advanced_mode(True)

            # Load settings from config JSON.
            advnc_mode.load_json(CAM_CONFIG_JSON)

            print(f"RealSense camera started with aligned color and depth streams "
                f"({self.width}x{self.height}@{self.fps}fps)")
        except Exception as e:
            print(f"[ERROR] Failed to initialize RealSense camera: {e}")
            raise  # Or use sys.exit(1) if running as a standalone script

    def get_color_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            return None
        # Return the depth frame (do not convert to numpy array so get_distance() remains available)
        return depth_frame

    def get_color_and_depth_frames(self) -> tuple:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        
        # Get intrinsics from the depth stream.
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        # Return the color image, the raw depth frame, and the intrinsics.
        return color_image, depth_frame, depth_intrinsics

    def get_color_intrinsics(self, depth_intrinsics) -> tuple:
        """
        Retrieve the color camera's intrinsic matrix and distortion coefficients.

        Returns:
            camera_matrix (np.ndarray): 3x3 intrinsic matrix.
            dist_coeffs (np.ndarray): Distortion coefficients array.
        """
        # Reuse our get_color_and_depth_frames to get the intrinsics
        
        if depth_intrinsics is None:
            return None, None
        # Build the 3x3 camera matrix from the intrinsics.
        camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                  [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        # Get the distortion coefficients.
        dist_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32)
        return camera_matrix, dist_coeffs

    def get_depth_intrinsics(self, depth_frame) -> tuple:
        """
        Retrieve the depth camera's intrinsic matrix and distortion coefficients.

        Returns:
            camera_matrix (np.ndarray): 3x3 intrinsic matrix.
            dist_coeffs (np.ndarray): Distortion coefficients array.
        """
        if not depth_frame:
            return None, None
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                  [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32)
        return camera_matrix, dist_coeffs

    def get_all_board_pose(self, aruco_dict):

        color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
        if color_image is None or depth_frame is None:
            print("Failed to capture color/depth.")
            return None, []
        
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        output_image = color_image.copy()

        marker_points = []
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers on the image.
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            
            # Process each detected marker.
            for i in range(len(ids)):
                c = corners[i][0]  # corners for marker i; shape (4,2)
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                # Draw the center for visualization.
                cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)
                
                # Get depth at the marker center.
                depth = depth_frame.get_distance(cx, cy)
                if depth > 0:
                    point_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                    # Depending on your coordinate conventions, you may adjust the axes.
                    x = point_coords[0]
                    y = point_coords[1]
                    z = point_coords[2]
                    point3d = Point3D(x, y, z)
                else:
                    print(f"Invalid depth at marker {i} (pixel: {cx},{cy}).")
                    point3d = Point3D(0, 0, 0)
                
                # Extract the marker id.
                marker_id = int(ids[i][0])
                marker_points.append({"id": marker_id, "point": point3d})
                # print(f"Marker id {marker_id} at pixel ({cx}, {cy}) -> 3D: {point3d}")
                
            return output_image, marker_points
        else:
            print("No markers detected.")
            return output_image, []
        
    def get_all_board_pose_v2(self, aruco_dict, captures: int = 5, delay_between: float = 0.5, max_round_retries: int = 3):
        """
        Enhanced multi-capture ArUco detection with IQR+median filtering per marker.
        Retries a round up to max_round_retries if not all markers from first round are detected
        and if samples are invalid. Ensures consistency of IDs across rounds.
        """
        samples: Dict[int, List[Point3D]] = {}
        last_image = None
        parameters = cv2.aruco.DetectorParameters()
        initial_ids = None
        print(f"[get_all_board_pose_v2] captures={captures}, delay={delay_between}, max_retries={max_round_retries}")
        # perform each capture round
        for round_idx in range(1, captures + 1):
            print(f" Round {round_idx}/{captures}")
            round_ids = None
            # retry loop for this round
            for attempt in range(1, max_round_retries + 1):
                color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
                last_image = color_image
                if color_image is None or depth_frame is None:
                    print(f"  [Attempt {attempt}] Frame capture failed.")
                    time.sleep(delay_between)
                    continue
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None or len(ids) == 0:
                    print(f"  [Attempt {attempt}] No markers.")
                    time.sleep(delay_between)
                    continue
                current_ids = set(int(i[0]) for i in ids)
                if initial_ids is None:
                    initial_ids = current_ids.copy()
                    print(f"  Initial marker set: {initial_ids}")
                else:
                    missing = initial_ids - current_ids
                    if missing:
                        print(f"  [Attempt {attempt}] Missing markers {missing}, retrying.")
                        time.sleep(delay_between)
                        continue
                # all markers present, collect samples
                vis = color_image.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                for idx, arr in enumerate(ids):
                    mid = int(arr[0])
                    c = corners[idx][0]
                    cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                    depth = depth_frame.get_distance(cx, cy)
                    if depth <= 0:
                        print(f"   [Attempt {attempt}] ID {mid} invalid depth.")
                        continue
                    xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                    pt = Point3D(*xyz)
                    if pt.x == 0 and pt.y == 0 and pt.z == 0:
                        print(f"   [Attempt {attempt}] ID {mid} zero sample, skipping.")
                        continue
                    print(f"   sample ID {mid}: {pt}")
                    samples.setdefault(mid, []).append(pt)
                # show this attempt's results
                cv2.imshow(f"cap{round_idx}_att{attempt}", vis)
                cv2.waitKey(200)
                cv2.destroyWindow(f"cap{round_idx}_att{attempt}")
                time.sleep(delay_between)
                break  # exit retry loop, move to next round
            else:
                print(f"  Round {round_idx} failed after {max_round_retries} attempts (incomplete marker set).")
        # Filter and assemble results
        results = []
        for mid, pts in samples.items():
            print(f"ID {mid} collected {len(pts)} samples")
            if not pts:
                continue
            robust_pt = self.get_coordinate_result_by_filter(pts)
            print(f" ID {mid} robust point: {robust_pt}")
            results.append({"id": mid, "point": robust_pt})
        output_image = last_image.copy() if last_image is not None else None
        if output_image is not None:
            gray2 = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            detector2 = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners_final, ids_final, _ = detector2.detectMarkers(gray2)
            if ids_final is not None and len(ids_final) > 0:
                cv2.aruco.drawDetectedMarkers(output_image, corners_final, ids_final)
            for entry in results:
                p = entry["point"]
                cv2.circle(output_image, (int(p.x), int(p.y)), 6, (0, 255, 0), 2)
        print(f"[get_all_board_pose_v2] done, {len(results)} markers")
        return output_image, results

    def get_board_pose(self, aruco_dict, max_depth_retries: int = 3):
        for attempt in range(max_depth_retries):
            color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
            if color_image is None or depth_frame is None:
                print("Failed to capture color/depth.")
                return None, Point3D(0, 0, 0)

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
            output_image = color_image.copy()

            if ids is None or len(ids) == 0:
                print(f"[Attempt {attempt+1}] No markers detected.")
                continue  # retry

            # draw markers & compute center pixel
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            all_cx = [int(c[:,0].mean()) for c in (c[0] for c in corners)]
            all_cy = [int(c[:,1].mean()) for c in (c[0] for c in corners)]
            cx, cy = int(sum(all_cx)/len(all_cx)), int(sum(all_cy)/len(all_cy))
            cv2.circle(output_image, (cx, cy), 4, (0,0,255), -1)

            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                print(f"[Attempt {attempt+1}] Invalid depth ({depth:.3f}), retrying…")
                continue  # retry
            
            # valid depth → deproject and return
            point_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
            point3d = Point3D(*point_coords)
            print(f"Detected board center at: {point3d}")
            return output_image, point3d

        # if we exit loop without a good reading:
        print(f"Failed after {max_depth_retries} attempts, returning zero point.")
        return output_image, Point3D(0, 0, 0)

    def get_coordinate_result_by_filter(self,
        points: List[Union[Point3D, Tuple[float, float, float]]]
    ) -> Point3D:
        """
        Given a list of 3D points, filter each axis by IQR then return
        the axis-wise medians as a new Point3D.
        """
        # split into separate lists
        xs, ys, zs = [], [], []
        for p in points:
            x, y, z = (p.x, p.y, p.z) if isinstance(p, Point3D) else p
            xs.append(x); ys.append(y); zs.append(z)

        # filter each axis
        xs_f = self.filter_iqr(xs)
        ys_f = self.filter_iqr(ys)
        zs_f = self.filter_iqr(zs)

        # take medians
        mx = self.get_median(xs_f)
        my = self.get_median(ys_f)
        mz = self.get_median(zs_f)

        return Point3D(mx, my, mz)

    def get_quartile(self, data: List[float], quartile_pos: float) -> float:
        """Compute the quartile (25% or 75%) by linear interpolation."""
        if not data:
            raise ValueError("Empty data for quartile")
        dataset = sorted(data)
        n = len(dataset)
        # percentile position in 1-based indexing
        q_pos = (n + 1) * quartile_pos
        # floor and ceil indices
        i = max(1, min(n, int(math.floor(q_pos))))
        j = max(1, min(n, i + 1))
        # values at those positions
        v_i = dataset[i - 1]
        v_j = dataset[j - 1]
        # fractional part
        frac = q_pos - i
        return v_i + frac * (v_j - v_i)

    def get_median(self, data: List[float]) -> float:
        """Return the exact or interpolated median."""
        if not data:
            raise ValueError("Empty data for median")
        dataset = sorted(data)
        n = len(dataset)
        mid = (n + 1) / 2
        i = int(math.floor(mid)) - 1
        j = i + 1
        if mid.is_integer():
            return dataset[i]
        else:
            # interpolate between floor and ceil
            return (dataset[i] + dataset[j]) / 2.0
        
    def filter_iqr(self, values: List[float], k: float = 1.5) -> List[float]:
        """Keep only those values within [Q1 - k·IQR, Q3 + k·IQR]."""
        if len(values) < 2:
            return values[:]
        q1 = self.get_quartile(values, 0.25)
        q3 = self.get_quartile(values, 0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        return [v for v in values if lower <= v <= upper]
        
    def stop(self):
            """
            Stop the RealSense pipeline.
            """
            if self.pipeline is None and self.imu_pipe is None:
                print("Pipeline is not initialized.")
                return
            if self.pipeline and self.imu_pipe:
                print("Stopping RealSense camera...")
                self.pipeline.stop()
                self.imu_pipe.stop()
                self.imu_pipe = None
                self.pipeline = None
                self.config = None
                self.profile = None
                self.align = None
                self.align_depth = None 
                print("RealSense camera stopped.")
            else:
                print("Pipeline is already stopped or not initialized.")
 
    def cam_capture_marker(self, aruco_dict_type):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose (camera coordinate system) as a Point3D instance.
        """
        # cv2.aruco.DICT_5X5_1000
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        image_marked, point3d = self.get_all_board_pose(aruco_dict)
        # print("Camera measurement:", point3d)
        if image_marked is not None:
            cv2.imshow("Detected Board", image_marked)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        
        return point3d, image_marked
    
    def cam_capture_marker_v2(self, aruco_dict_type):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose (camera coordinate system) as a Point3D instance.
        """
        # cv2.aruco.DICT_5X5_1000
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        image_marked, point3d = self.get_all_board_pose_v2(aruco_dict)
        # print("Camera measurement:", point3d)
        if image_marked is not None:
            cv2.imshow("Detected Board", image_marked)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        
        return point3d, image_marked
  
    def save_image(self, image, path):
        """
        Saves the given image to the specified path with a timestamp as the name.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(path, f"{timestamp}.png")
        cv2.imwrite(full_path, image)
        print(f"Marked image saved to {full_path}")
 
    def cam_capture_marker_jupyter(self, aruco_dict_type):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose (camera coordinate system) as a Point3D instance.
        """

        # cv2.aruco.DICT_5X5_1000
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        image_marked, point3d = self.get_all_board_pose(aruco_dict)
        # print("Camera measurement:", point3d)
        if image_marked is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(image_marked, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        return point3d
    
    def load_matrix(self):
        """
        Loads the transformation matrix from a file.
        Returns the transformation matrix.
        """
        try:
            with open(CONFIG_MATRIX_PATH, 'r') as f:
                loaded_data = json.load(f)
                name = loaded_data["name"]
                print("Loaded matrix name:", name)
                # Convert the list to a numpy array.
                self.best_matrix = np.array(loaded_data["matrix"])

        except FileNotFoundError:
            print("Transformation matrix file not found.")
            return None

    def transform_marker_points(self, marker_points):
        """
        Applies a 4x4 transformation matrix to a list of marker points.
        
        Parameters:
        - marker_points: A list of dictionaries, each in the format:
                {"id": <marker_id>, "point": Point3D(x, y, z)}
        - transformation_matrix: A 4x4 numpy array representing the transformation.
        
        Returns:
        - transformed_points: A list of dictionaries, each with the marker id and
            its transformed point as a Point3D.
        """
        if self.best_matrix is None:
            print("No transformation matrix loaded.")
            return None
        
        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            # Create the homogeneous coordinate (4,1) by appending a 1.
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            # Multiply by the transformation matrix.
            transformed_homo = self.best_matrix @ homo_pt
            # Convert back to Cartesian coordinates (assuming transformation_matrix is affine).
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]  # in case scale != 1
            # Create a new Point3D object for the transformed point.
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points

    def restart(self):
        """
        Restart the RealSense pipeline.
        """
        if self.pipeline is None:
            print("Pipeline is not initialized. Cannot restart.")
            return
        print("Restarting RealSense camera...")
        # Stop the pipeline before reinitializing
        self.stop()
        self.init_cam()
        print("RealSense camera restarted with aligned color and depth streams.")

    def __del__(self):
        self.stop()

