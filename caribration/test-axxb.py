import cv2
import numpy as np
import time
from classrobot import axxb_cam  
from classrobot.point3d import Point3D
from scipy.spatial.transform import Rotation as R

def main():
    cam = axxb_cam.RealsenseCam(width=640, height=480, fps=30)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    marker_length = 0.1  # for a 100x100 mm marker
    
    image_marked, point3d, rotation_matrix = cam.get_board_pose_with_rotation(aruco_dict, marker_length)
    
    if image_marked is not None:
        cv2.imshow("Detected Board with Corrected Axes", image_marked)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        
        print("Detected Translation (tvec):")
        print(f"X: {point3d.x}, Y: {point3d.y}, Z: {point3d.z}")
        if rotation_matrix is not None:
            print("Detected Rotation Matrix:")
            print(rotation_matrix)
        else:
            print("No rotation matrix detected.")
    else:
        print("No image captured or marker not detected.")
        
    cam.stop()

if __name__ == "__main__":
    main()
