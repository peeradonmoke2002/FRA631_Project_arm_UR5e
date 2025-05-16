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


    def _capture_frames(self) -> Tuple[np.ndarray, rs.depth_frame, rs.intrinsics]:
        """
        Capture and align color and depth frames.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("Failed to capture aligned frames.")
        intr = depth.profile.as_video_stream_profile().intrinsics
        img = np.asanyarray(color.get_data())
        return img, depth, intr


    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (camera_matrix, dist_coeffs)."""
        _, _, intr = self._capture_frames()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        D = np.array(intr.coeffs, dtype=float)
        return K, D
    
    # --- ArUco Marker Detection ---

    def detect_all_markers(
        self, 
        aruco_dict
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Point3D]]]:

        try:
            img, depth, intrinsics = self._capture_frames()
        except Exception as e:
            print(f"[detect_all_markers] Frame error: {e}")
            return None, []
        # Check the number of channels in the image
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA image
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        annotated_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or not ids.size:
            print("[detect_all_markers] No markers found.")
            return annotated_img, []

        cv2.aruco.drawDetectedMarkers(annotated_img, corners, ids)

        results: List[Dict[str, Point3D]] = []
        for corner, id_arr in zip(corners, ids):
            marker_id = int(id_arr[0])
            cx, cy = map(int, corner[0].mean(axis=0))
            cv2.circle(annotated_img, (cx, cy), 4, (0, 0, 255), -1)

            depth_val = depth.get_distance(cx, cy)
            if depth_val > 0:
                x, y, z = rs.rs2_deproject_pixel_to_point(
                    intrinsics, [cx, cy], depth_val
                )
                pt = Point3D(x, y, z)
            else:
                print(f"[detect_all_markers] Marker {marker_id} invalid depth at ({cx},{cy}).")
                pt = Point3D(0, 0, 0)

            results.append({"id": marker_id, "point": pt})

        return annotated_img, results


    def detect_markers_multi(
        self,
        aruco_dict,
        rounds: int = 5,
        delay: float = 0.5,
        max_tries: int = 3
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Point3D]]]:

        samples: Dict[int, List[Point3D]] = {}
        annotated_img = None
        # print(f"[detect_markers_multi] rounds={rounds}, delay={delay}, max_tries={max_tries}")

        initial_ids = None
        params = cv2.aruco.DetectorParameters()
        
        for round_idx in range(1, rounds + 1):
            # print(f"[detect_markers_multi] Round {round_idx}/{rounds}")
            for attempt in range(1, max_tries + 1):
                try:
                    img, depth, intr = self._capture_frames()
                except Exception as e:
                    print(f"  [Attempt {attempt}] frame error: {e}")
                    time.sleep(delay)
                    continue
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # RGBA image
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    
                annotated_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(gray)

                if ids is None or len(ids) == 0:
                    print(f"  [Attempt {attempt}] no markers, retrying…")
                    time.sleep(delay)
                    continue

                current_ids = {int(i[0]) for i in ids}
                if initial_ids is None:
                    initial_ids = current_ids.copy()
                    print(f"  Initial IDs: {initial_ids}")
                elif initial_ids != current_ids:
                    missing = initial_ids - current_ids
                    print(f"  [Attempt {attempt}] missing {missing}, retrying…")
                    time.sleep(delay)
                    continue

                # all markers present: collect samples
                for corner, id_arr in zip(corners, ids):
                    mid = int(id_arr[0])
                    cx, cy = map(int, corner[0].mean(axis=0))
                    d = depth.get_distance(cx, cy)
                    if d <= 0:
                        print(f"   [Attempt {attempt}] ID {mid} invalid depth")
                        continue
                    x, y, z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], d)
                    pt = Point3D(x, y, z)
                    samples.setdefault(mid, []).append(pt)
                    # print(f"   sample ID {mid}: {pt}")

                # done with this round
                time.sleep(delay)
                break
            else:
                print(f"  Round {round_idx} failed after {max_tries} tries")

        # compute robust centers
        results: List[Dict[str, Point3D]] = []
        for mid, pts in samples.items():
            if not pts:
                continue
            robust = self.get_coordinate_result_by_filter(pts)
            # print(f"[detect_markers_multi] ID {mid} → {robust}")
            results.append({"id": mid, "point": robust})

        # annotate final image
        if annotated_img is not None:
            cv2.aruco.drawDetectedMarkers(annotated_img, corners, ids)
            for entry in results:
                p = entry["point"]
                cv2.circle(annotated_img, (int(p.x), int(p.y)), 6, (0, 255, 0), 2)

        # print(f"[detect_markers_multi] done, {len(results)} markers found")
        return annotated_img, results

    def get_single_board_pose(self, aruco_dict, max_tries: int = 3):
        """
        Try up to max_tries to detect ArUco markers, compute their average 3D center,
        and return (annotated_image, Point3D). If all attempts fail, returns
        (last_captured_image or None, Point3D(0, 0, 0)).
        """
        last_image = None

        for idx in range(1, max_tries + 1):
            # 1. grab frames
            try:
                img, depth, intr = self._capture_frames()
            except Exception as e:
                print(f"[get_single_board_pose] Try {idx}: frame error ({e})")
                continue
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA image
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            last_image = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. detect markers
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                print(f"[get_single_board_pose] Try {idx}: no markers found")
                continue

            # draw markers
            cv2.aruco.drawDetectedMarkers(last_image, corners, ids)

            # 3. find pixel center of all markers
            centers = [c[0].mean(axis=0) for c in corners]  # list of [x, y]
            cx, cy = map(int, np.mean(centers, axis=0))
            cv2.circle(last_image, (cx, cy), 5, (0, 0, 255), -1)

            # 4. get depth
            dist = depth.get_distance(cx, cy)
            if dist <= 0:
                print(f"[get_single_board_pose] Try {idx}: bad depth {dist:.3f}, retrying")
                continue

            # 5. deproject and return
            xyz = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
            return last_image, Point3D(*xyz)

        # all tries exhausted
        print("[get_single_board_pose] All attempts failed—returning fallback.")
        return last_image, Point3D(0, 0, 0)

    def get_coordinate_result_by_filter(self,
        points: List[Union[Point3D, Tuple[float, float, float]]]
    ) -> Point3D:
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
        if len(values) < 2:
            return values[:]
        q1 = self.get_quartile(values, 0.25)
        q3 = self.get_quartile(values, 0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        return [v for v in values if lower <= v <= upper]
        
    def cam_capture_marker(self, aruco_dict_type):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose (camera coordinate system) as a Point3D instance.
        """
        # cv2.aruco.DICT_5X5_1000
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        image_marked, point3d = self.detect_all_markers(aruco_dict)
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
        image_marked, point3d = self.detect_markers_multi(aruco_dict)
        # print("Camera measurement:", point3d)
        if image_marked is not None:
            cv2.imshow("Detected Board", image_marked)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        
        return point3d, image_marked
  
    def save_image(self, image, path):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(path, f"{timestamp}.png")
        cv2.imwrite(full_path, image)
        print(f"Marked image saved to {full_path}")
 
    def cam_capture_marker_jupyter(self, aruco_dict_type):
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

# --- restart & stop  

    def restart(self):
        if self.pipeline is None:
            print("Pipeline is not initialized. Cannot restart.")
            return
        print("Restarting RealSense camera...")
        # Stop the pipeline before reinitializing
        self.stop()
        self.init_cam()
        print("RealSense camera restarted with aligned color and depth streams.")

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
    

