import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

def estimate_transform_kabsch(source_pts: np.ndarray, target_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the rigid transformation (rotation and translation) that maps
    source_pts (board points) to target_pts (3D image points) using the Kabsch algorithm.
    
    Parameters:
        source_pts (np.ndarray): Nx3 array of 3D points in the board coordinate system.
        target_pts (np.ndarray): Nx3 array of corresponding 3D points in the camera coordinate system.
        
    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    assert source_pts.shape == target_pts.shape, "Point sets must have the same shape"
    # Compute centroids.
    centroid_source = np.mean(source_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    # Subtract centroids.
    src_centered = source_pts - centroid_source
    tgt_centered = target_pts - centroid_target
    # Compute covariance matrix.
    H = src_centered.T @ tgt_centered
    # Compute SVD.
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure right-handed coordinate system.
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source
    return R, t

class RealsenseCam:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the RealSense camera pipeline with both color and depth streams.
        
        Parameters:
            width (int): Width of the streams in pixels.
            height (int): Height of the streams in pixels.
            fps (int): Frame rate.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Enable both color and depth streams.
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)
        # Create an align object to align depth frames to the color frame.
        self.align = rs.align(rs.stream.color)
        print("RealSense camera started with aligned color and depth streams.")

    def get_color_frame(self) -> np.ndarray:
        """
        Capture a single color frame from the aligned frames.
        
        Returns:
            image (np.ndarray): Captured BGR image, or None if no frame is available.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_depth_frame(self) -> np.ndarray:
        """
        Capture a single depth frame from the aligned frames.
        
        Returns:
            depth_image (np.ndarray): Captured depth image, or None if no frame is available.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            return None
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def get_color_and_depth_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture both a color and a depth frame from the aligned frames.
        
        Returns:
            color_image (np.ndarray): Captured BGR image.
            depth_image (np.ndarray): Captured depth image.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def get_board_pose(self, board, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """
        Capture aligned color and depth frames, detect the ChArUco board, and compute the board center in 3D.
        
        This method uses depth data to unproject detected 2D charuco corners into 3D points,
        estimates a rigid transformation via the Kabsch algorithm using the board object points corresponding to the detected corners,
        and then computes the board center (as the average of all defined board corners in the board coordinate system)
        transformed into the camera coordinate system.
        
        Parameters:
            board: An instance of cv2.aruco.CharucoBoard.
            camera_matrix (np.ndarray): Camera intrinsic parameters.
            dist_coeffs (np.ndarray): Distortion coefficients.
        
        Returns:
            board_center_camera (np.ndarray): 3x1 array representing the x, y, z position of the board center in camera coordinates.
            Returns None if detection or pose estimation fails.
        """
        # Get aligned color and depth frames.
        color_image, depth_image = self.get_color_and_depth_frames()
        if color_image is None or depth_image is None:
            print("Failed to capture both color and depth images.")
            return None
        
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        # Detect ArUco markers.
        corners, ids, _ = aruco.detectMarkers(gray, board.dictionary, parameters=parameters)
        if ids is None or len(ids) < 4:
            print("Not enough markers detected.")
            return None

        # Interpolate ChArUco corners.
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is None or len(charuco_corners) < 4:
            print("Not enough ChArUco corners detected.")
            return None

        # Prepare lists for 3D image points and corresponding board object points (from detected corners).
        img_points_3d = []
        board_obj_points = []
        # Camera intrinsics.
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        for i, corner in enumerate(charuco_corners):
            # Get pixel coordinates.
            u, v = corner.ravel()
            u_int, v_int = int(round(u)), int(round(v))
            depth_value = depth_image[v_int, u_int]
            if depth_value == 0:
                continue
            # Convert depth from millimeters to meters.
            depth_m = depth_value / 1000.0
            # Unproject pixel to 3D.
            X = (u - cx) * depth_m / fx
            Y = (v - cy) * depth_m / fy
            Z = depth_m
            img_points_3d.append([X, Y, Z])
            
            # Get corresponding board object point.
            # charuco_ids gives the index corresponding to board.chessboardCorners.
            board_idx = int(charuco_ids[i])
            board_pt_2d = board.chessboardCorners[board_idx].ravel()  # [x, y]
            board_obj_points.append([board_pt_2d[0], board_pt_2d[1], 0.0])
        
        if len(img_points_3d) < 4 or len(board_obj_points) < 4:
            print("Not enough valid 3D points for pose estimation.")
            return None
        
        img_points_3d = np.array(img_points_3d, dtype=np.float32)
        board_obj_points = np.array(board_obj_points, dtype=np.float32)
        
        # Estimate the transformation (R, t) mapping board object points to 3D image points.
        R, t = estimate_transform_kabsch(board_obj_points, img_points_3d)
        
        # Compute board center in board coordinate system using all defined board corners.
        board_all = board.chessboardCorners.reshape(-1, 2)  # shape: (N, 2)
        board_center_2d = np.mean(board_all, axis=0)         # average x, y
        board_center_board = np.array([board_center_2d[0], board_center_2d[1], 0.0], dtype=np.float32)
        
        # Transform board center into camera coordinate system.
        board_center_camera = R.dot(board_center_board) + t
        
        return board_center_camera

    def stop(self):
        """
        Stop the RealSense pipeline.
        """
        self.pipeline.stop()
        print("RealSense camera stopped.")
