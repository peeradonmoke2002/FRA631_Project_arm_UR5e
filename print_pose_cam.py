# print_pose_cam.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import realsense_cam
from classrobot.point3d import Point3D


class PrintPoseCam():
    def __init__(self):
        # HOME position of the robot
        self.HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
                         -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        # fix Y position and roll-pitch-yaw
        self.FIX_Y = 0.18427318897339476
        self.RPY_HOME = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.RPY_GRAPING = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
        # Load the transformation matrix from a JSON file.
        path_matrix = "FRA631_Project_Dual_arm_UR5_Calibration/caribration/config/best_matrix.json"
        self.config_matrix_path = pathlib.Path(__file__).parent / path_matrix
        self.cam = realsense_cam.RealsenseCam()
        self.best_matrix = self.load_matrix()

    def stop_all(self):
        self.cam.stop()

    def cam_relasense(self):
        """
        Launches the RealSense camera to obtain the board pose.
        Returns the board pose (camera coordinate system) as a Point3D instance.
        """

        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
        image_marked, point3d = self.cam.get_all_board_pose(aruco_dict)
        # print("Camera measurement:", point3d)
        if image_marked is not None:
            cv.imshow("Detected Board", image_marked)
            cv.waitKey(5000)
            cv.destroyAllWindows()
        return point3d
    
    def load_matrix(self):
        """
        Loads the transformation matrix from a file.
        Returns the transformation matrix.
        """
        try:
            with open(self.config_matrix_path, 'r') as f:
                loaded_data = json.load(f)
                name = loaded_data["name"]
                best_matrix = np.array(loaded_data["matrix"])
                return best_matrix
        except FileNotFoundError:
            print("Transformation matrix file not found.")
            return None


    def transform_marker_points(self,marker_points, transformation_matrix):
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

        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            # Create the homogeneous coordinate (4,1) by appending a 1.
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            # Multiply by the transformation matrix.
            transformed_homo = transformation_matrix @ homo_pt
            # Convert back to Cartesian coordinates (assuming transformation_matrix is affine).
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]  # in case scale != 1
            # Create a new Point3D object for the transformed point.
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points


 
    def printPoseCam(self):
        if self.best_matrix is None:
            print("Failed to load transformation matrix.")
            return
        
        maker_point = self.cam_relasense()
        print(maker_point)
        transfrom_point = self.transform_marker_points(maker_point, self.best_matrix)
        print(transfrom_point)
        # Sort the markers by their id in ascending order.
        sorted_markers = sorted(transfrom_point, key=lambda m: m["id"])
    
        for marker in sorted_markers:
            marker_id = marker["id"]
            point = marker["point"]  # This is an instance of Point3D.
            print(marker_id)
            print(point.x, point.y, point.z)    
         
        
def main():
    try:
        printPoseCam = PrintPoseCam()
        printPoseCam.printPoseCam()
        printPoseCam.stop_all()
        print("Finished")

    except Exception as e:
        print(f"Error initializing Move2Object: {e}")
        return


if __name__ == "__main__":
    main()
