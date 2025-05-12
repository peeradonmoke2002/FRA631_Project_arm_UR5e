# print_pose_cam.py
import sys
import pathlib
import cv2
from typing import List, Dict

# add parent directory to path so you can import your module
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from classrobot.realsense_cam import RealsenseCam
from classrobot.point3d import Point3D


class PrintPoseCam:
    def __init__(self):
        # initialize your camera
        self.cam = RealsenseCam()

    def stop_all(self):
        """Shut down the RealSense pipelines cleanly."""
        self.cam.stop()

    def capture_markers(self) -> List[Dict[str, Point3D]]:
        """
        Capture a single frame, detect all markers, and return
        the raw list of {'id': ..., 'point': Point3D}.
        """
        # This returns (marker_list, image)
        marker_list, _ = self.cam.cam_capture_marker_v2(cv2.aruco.DICT_5X5_1000)
        return marker_list or []

    def print_pose_cam(self):
        """
        Main logic: get marker points, transform them, sort, and print.
        """
        raw_markers = self.capture_markers()
        if not raw_markers:
            print("No markers detected.")
            return

        # transform to robot coords
        transformed = self.cam.transform_marker_points(raw_markers)
        if not transformed:
            print("No transformation matrix loaded or error transforming.")
            return

        # sort by ID
        sorted_markers = sorted(transformed, key=lambda m: m["id"])

        for m in sorted_markers:
            mid = m["id"]
            pt: Point3D = m["point"]
            # print(f"Marker {mid}: x={pt.x:.3f}, y={pt.y:.3f}, z={pt.z:.3f}")

def main():
    printer = PrintPoseCam()
    try:
        printer.print_pose_cam()
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        printer.stop_all()
        print("Camera stopped. Exiting.")

if __name__ == "__main__":
    main()
