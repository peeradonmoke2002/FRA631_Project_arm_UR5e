import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

import numpy as np

def estimate_transform_kabsch(source_pts, target_pts):
    """
    Estimate the rigid transformation (rotation and translation) that maps
    source_pts to target_pts using the Kabsch algorithm.

    Parameters:
        source_pts (array-like): Nx3 array of points in the source coordinate system.
        target_pts (array-like): Nx3 array of corresponding points in the target coordinate system.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    # Convert inputs to numpy arrays if they are not already.
    source_pts = np.asarray(source_pts, dtype=np.float32)
    target_pts = np.asarray(target_pts, dtype=np.float32)
    
    # Ensure both point sets have the same shape.
    assert source_pts.shape == target_pts.shape, "Point sets must have the same shape"
    
    # Compute centroids of both point sets.
    centroid_source = np.mean(source_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    
    # Center the points around their centroids.
    src_centered = source_pts - centroid_source
    tgt_centered = target_pts - centroid_target
    
    # Compute the covariance matrix.
    H = np.dot(src_centered.T, tgt_centered)
    
    # Compute SVD of the covariance matrix.
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix.
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system; if not, adjust.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute the translation vector.
    t = centroid_target - np.dot(R, centroid_source)
    
    return R, t.reshape(3, 1)


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

    def get_board_pose(self, board, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture aligned color and depth frames, detect the ChArUco board using the new API,
        compute the board center in 3D, and return a marked image showing detections.

        Parameters:
            board: An instance of cv2.aruco.CharucoBoard.
            camera_matrix (np.ndarray): Camera intrinsic parameters.
            dist_coeffs (np.ndarray): Distortion coefficients.
        
        Returns:
            board_center_camera (np.ndarray): 3x1 array representing the x, y, z position of the board center 
                                            in camera coordinates.
            image_marked (np.ndarray): Color image with detected markers, corners, and the projected board center drawn.
            Returns (None, None) if detection or pose estimation fails.
        """
        # Get aligned color and depth frames.
        color_image, depth_image = self.get_color_and_depth_frames()
        if color_image is None or depth_image is None:
            print("Failed to capture both color and depth images.")
            return None, None

        # Convert to grayscale for detection.
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Use the new API: Create a CharucoDetector instance and detect the board.
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        newImage = color_image.copy()
        # Draw detected markers and corners on the image.
        # cv2.aruco.drawDetectedMarkers(newImage, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(newImage, charuco_corners, charuco_ids)
        
        if charuco_ids is None or len(charuco_ids) < 4:
            print("Not enough ChArUco corners detected.")
            return None, None
        
        return newImage



        




    def stop(self):
        """
        Stop the RealSense pipeline.
        """
        self.pipeline.stop()
        print("RealSense camera stopped.")
