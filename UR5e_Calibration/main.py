# main.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import math
from typing import List, Tuple, Union, Optional

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
import os

PATH_IMAGE_LOGS =  pathlib.Path(__file__).resolve().parent.parent / "images/calibration/logs"


class calibrationUR5e():
    def __init__(self):
        # End effector home position (6 DOF) and other test positions
        self.HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568, -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        self.robot_ip = "192.168.200.10"
        self.speed = 0.06
        self.acceleration = 0.20
        # Initialize the robot connection once.
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)
        self.cam = realsense_cam.RealsenseCam()

    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()

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

    def cam_relasense(self,
                    captures: int = 5,
                    max_retry: int = 5,
                    retry_delay: float = 0.25,
                    inter_capture_delay: float = 2.0) -> Optional[Point3D]:
        """
        Capture 'captures' valid board poses, then filter them by IQR per axis
        and return the median Point3D. Each individual capture will retry
        up to max_retry times if detection/depth fails.
        """
        pts: List[Point3D] = []
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        last_img = None

        # 1) Collect 'captures' non-zero readings
        while len(pts) < captures:
            for attempt in range(1, max_retry + 1):
                img, p = self.cam.get_single_board_pose(aruco_dict)
                last_img = img
                # treat (0,0,0) as invalid
                if p is not None and any((p.x, p.y, p.z)):
                    pts.append(p)
                    break
                print(f"  Capture {len(pts)+1}, attempt {attempt}/{max_retry} failed.")
                time.sleep(retry_delay)
            else:
                print(f" → Couldn’t get capture #{len(pts)+1} after {max_retry} tries.")
                break

            # wait before next capture
            print(f"  Capture {len(pts)}/{captures} successful.")
            print(f"  Pose: {pts[-1]}")
            time.sleep(inter_capture_delay)

        if not pts:
            print("No valid poses collected.")
            return None

        # 2) Filter + median – use your Python port of GetCoordinateResultByFilter
        robust_pt = self.get_coordinate_result_by_filter(pts)  # returns a Point3D

        # 3) (Optional) show & save the last image
        if last_img is not None:
            cv.imshow("Detected Board (filtered)", last_img)
            cv.waitKey(5000)
            cv.destroyAllWindows()
            # self.cam.save_image(last_img, PATH_IMAGE_LOGS)

        print(f"Robust board pose after IQR filter: {robust_pt}")
        return robust_pt

    
    def save_images(self,image):
        self.cam.save_image(image, PATH_IMAGE_LOGS)
    
    def get_robot_TCP(self):
        """
        Connects to the robot and retrieves the current TCP (end-effector) position.
        Returns a 3-element list: [x, y, z].
        """
        pos = self.robot.robot_get_position()
        pos_3d = self.robot.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS, self.speed, self.acceleration)

    def moveL_square(self):
   
        # ----- Reference data -----
        # Use the provided "home" and reference corner positions.
        pos_home = [0.7011797304915488, 0.18427154391614353, 0.17217411213036665]
        BL_ref = [0.6158402179629584, 0.18426921633782534, 0.3510680386492011]
        TL_ref = [0.9034970156209872, 0.18431919874933683, 0.3510680386492011]
        TR_ref = [0.9035034184486379, 0.18425659123476879, -0.43708867396716417]
        BR_ref = [0.6158402179629584, 0.18424774957164802, -0.43708867396716417]
        RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        avg_y = (TL_ref[1] + TR_ref[1] + BL_ref[1] + BR_ref[1]) / 4.0
        
        # ----- Build a 3x3 grid on the x-z plane via bilinear interpolation -----
        grid = []  # grid[i][j] corresponds to position at row i, col j.
        for i in range(3):
            row_positions = []
            v = i / 2.0  # v = 0.0, 0.5, 1.0 for rows 0,1,2
            for j in range(3):
                u = j / 2.0  # u = 0.0, 0.5, 1.0 for columns 0,1,2
                # Bilinear interpolation: x and z; y is fixed to avg_y.
                x = (1 - u) * (1 - v) * TL_ref[0] + u * (1 - v) * TR_ref[0] + (1 - u) * v * BL_ref[0] + u * v * BR_ref[0]
                z = (1 - u) * (1 - v) * TL_ref[2] + u * (1 - v) * TR_ref[2] + (1 - u) * v * BL_ref[2] + u * v * BR_ref[2]
                row_positions.append([x, avg_y, z])
            grid.append(row_positions)

        move_order = [
            (2, 0),  # Home: bottom-left
            (2, 1),  # bottom-center
            (2, 2),  # bottom-right
            (1, 2),  # middle-right
            (0, 2),  # top-right
            (0, 1),  # top-center
            (0, 0),  # top-left
            (1, 0),  # middle-left
            (1, 1)   # center
        ]
        
        # ----- Outer loop: iterate over 3 vertical planes -----
        # For each plane, we adjust the y coordinate relative to the original grid.
        vertical_distance = 0.05  # 10 cm per plane
        num_planes = 4
        
        # For state numbering, we can use: state = plane_idx * 10 + grid_index + 1
        for plane_idx in range(num_planes):
            # For each plane, adjust the grid: the new y becomes avg_y - (plane_idx * vertical_distance)
            plane_offset_y = avg_y - (plane_idx * vertical_distance)
            print(f"--- Moving on plane {plane_idx+1} with y = {plane_offset_y:.4f} ---")
            
            # For this plane, update the grid positions (only y changes)
            plane_grid = []
            for i in range(3):
                row_positions = []
                for j in range(3):
                    pos = grid[i][j].copy()
                    pos[1] = plane_offset_y
                    row_positions.append(pos)
                plane_grid.append(row_positions)
            
            # Iterate over the move order for this plane.
            for idx, (i, j) in enumerate(move_order):
                pos = plane_grid[i][j]
                # Combine position and fixed orientation (RPY) into the target.
                target = pos + RPY
                state = plane_idx * 10 + idx + 1
                print(f"Moving to plane {plane_idx+1} grid cell ({i},{j}) - Target: {target}")
                self.robot.robot_moveL(target, self.speed, self.acceleration)
                time.sleep(6)  # Pause for the robot to settle
                
                if self.collect_data(state=state):
                    print(f"Data collection successful for state {state}.")
                else:
                    print(f"Data collection failed for state {state}. Halting sequence.")
                    # self.robot.robot_release()
                    return

    def collect_data(self, state: int) -> bool:
        """
        Collect calibration data at the current robot state.
        This function collects:
           - the board pose from the camera (ccs: camera coordinate system)
           - the robot TCP (ac: actual coordinate, i.e., end-effector)
        It then appends a row to a CSV file in the format:
           Pos, ccs_x, ccs_y, ccs_z, ac_x, ac_y, ac_z
        Parameters:
            state (int): A state number or position index.
        Returns:
            bool: True if data collection is successful.
        """
        # filename = os.path.join(os.path.dirname(__file__), "data", "calibration_data.csv")
        filename =  pathlib.Path(__file__).resolve().parent.parent / "data/calibration_data.csv"

        print(f"Collecting data for state {state} ...")


        # data collection ------
        ccs = self.cam_relasense()  # This should return a Point3D object
        ccs = ccs.to_list()
        # Get robot TCP (ac)
        ac = self.get_robot_TCP()  # This returns a list [x, y, z] maker
        # -- end data collection ------


        # Create a CSV row.
        data_row = [state, ccs[0], ccs[1], ccs[2], ac[0], ac[1], ac[2]]
        print("Collected data row:", data_row)
        
        # Check if file exists; if not, write header.
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write("Pos,ccs_x,ccs_y,ccs_z,ac_x,ac_y,ac_z\n")
            # Write data row.
            f.write(",".join(map(str, data_row)) + "\n")
        return True

def main():
    calibration = calibrationUR5e()

    calibration.move_home()
    time.sleep(3)
    calibration.moveL_square()
    time.sleep(3)
    calibration.move_home()
    calibration.stop_all()

if __name__ == "__main__":
    main()
