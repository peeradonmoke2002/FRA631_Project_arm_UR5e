import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json

# Append the parent directory to sys.path เพื่อให้สามารถ import โมดูลแบบ relative ได้
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
from classrobot import gripper

class Move2Object():
    def __init__(self):
        # Home position และพารามิเตอร์พื้นฐาน
        self.HOME_POS = [
            0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7318488600590023, 0.686830145115122, -1.731258978679887
        ]
        # ปรับ IP ให้ตรงกับระบบของคุณ
        self.robot_ip = "172.17.0.2"
        self.speed = 0.1
        self.acceleration = 1.2

        # ค่า fix Y (ในกรณีที่ต้องการคงค่า Y จากฐานหุ่นยนต์ไว้)
        self.FIX_Y = 0.18427318897339476
        # ค่า RPY ที่ได้จากการคำนวณ (หรือค่าที่กำหนดไว้ล่วงหน้า)
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        # ค่า Test_RPY สำหรับการสั่งเคลื่อนที่แบบคงที่
        self.Test_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]

        # กำหนด path ของ transformation matrix (best_matrix.json)
        self.config_matrix_path = pathlib.Path("/home/tang/ur_robot/FRA631_Project_Dual_arm_UR5_planing/config/best_matrix.json")


        # สร้างการเชื่อมต่อกับหุ่นยนต์ผ่าน RTDE
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)

        # สำหรับตัวอย่างนี้ แม้ว่าเราจะไม่ใช้กล้องจริง 

        # โหลด transformation matrix จากไฟล์ best_matrix.json
        self.best_matrix = self.load_matrix()




    def cam_relasense(self):
        """
        จำลองการรับ marker data จากกล้อง (camera coordinate system)
        คืนค่าเป็น list ของ dictionary แต่ละตัวมี 'id' และ 'point' (Point3D)
        """
        return [
            {'id': 100, 'point': Point3D(0.388, 0.173, 1.509)},
            {'id': 102, 'point': Point3D(0.178, 0.169, 1.520)},
            {'id': 101, 'point': Point3D(0.282, 0.174, 1.512)},
            {'id': 14,  'point': Point3D(0.251, -0.061, 1.494)},
            {'id': 3,   'point': Point3D(-0.095, -0.050, 1.466)}
        ]
    
    def load_matrix(self):
        """
        Loads the transformation matrix from a file.
        Returns the transformation matrix (numpy array) or None if not found.
        """
        try:
            with open(self.config_matrix_path, 'r') as f:
                loaded_data = json.load(f)
                best_matrix = np.array(loaded_data["matrix"])
                return best_matrix
        except FileNotFoundError:
            print("Transformation matrix file not found.")
            return None

    def get_robot_TCP(self):
        """
        สอบถามตำแหน่ง TCP (Tool Center Point) ของหุ่นยนต์
        คืนค่าเป็น list [x, y, z]
        """
        pos = self.robot.robot_get_position()
        pos_3d = self.robot.convert_gripper_to_maker(pos)
        print("Robot TCP position:", pos_3d)
        return pos_3d

    def move_home(self):
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS, self.speed)

    def transform_marker_points(self, marker_points, transformation_matrix):
        """
        นำ marker data (แต่ละ marker มี 'id' และ 'point' เป็น Point3D)
        มาคูณกับ transformation matrix (best_matrix) เพื่อแปลง coordinate 
        จากระบบกล้องไปยังระบบฐานหุ่นยนต์

        Parameters:
            marker_points: list of dictionaries ในรูปแบบ {"id": marker_id, "point": Point3D(x,y,z)}
            transformation_matrix: 4x4 numpy array ที่ได้จาก best_matrix.json

        Returns:
            transformed_points: list ของ dictionary โดยแต่ละตัวมี 'id' และ 'point' (Point3D ใหม่)
        """
        transformed_points = []
        for marker in marker_points:
            marker_id = marker["id"]
            pt = marker["point"]
            # สร้าง homogeneous coordinate (4,1) จาก [x, y, z] โดยเพิ่ม 1 ทางด้านท้าย
            homo_pt = np.array([pt.x, pt.y, pt.z, 1], dtype=np.float32).reshape(4, 1)
            # คูณกับ transformation matrix ที่ได้จาก best_matrix.json
            transformed_homo = transformation_matrix @ homo_pt
            # แปลงกลับเป็น Cartesian coordinate โดยหารด้วยค่า w (ส่วนที่ 4)
            transformed_pt = transformed_homo[:3, 0] / transformed_homo[3, 0]
            # สร้าง Point3D ใหม่จากค่า x, y, z ที่ได้
            transformed_point = Point3D(transformed_pt[0], transformed_pt[1], transformed_pt[2])
            transformed_points.append({"id": marker_id, "point": transformed_point})
        return transformed_points


    def move_muti_to_object(self):
        """
        ขั้นตอนการสั่งหุ่นยนต์เคลื่อนที่ตาม marker data:
          1. รับ marker data จำลอง (จาก cam_relasense())
          2. แปลง marker data ด้วย transformation matrix (best_matrix.json)
             ซึ่งจะเปลี่ยนค่า Point3D จากระบบกล้องเป็นระบบฐานหุ่นยนต์
          3. จัดเรียง marker ตาม id (ascending)
          4. สำหรับแต่ละ marker สร้าง pose โดยใช้ค่า x, z ที่แปลงแล้ว แต่ fix ค่า y ด้วย self.FIX_Y
             จากนั้นรวมกับค่า RPY (Test_RPY)
          5. สั่งหุ่นยนต์เคลื่อนที่ไปยัง pose ที่กำหนด
        """
        if self.best_matrix is None:
            print("Failed to load transformation matrix.")
            return
        
        # 1. รับ marker data จำลอง
        marker_points = self.cam_relasense()
        print("Simulated marker data:")
        for marker in marker_points:
            pt = marker["point"]
            print(f"Marker {marker['id']}: ({pt.x}, {pt.y}, {pt.z})")
        
        # 2. แปลง marker data โดยใช้ best_matrix
        transformed_points = self.transform_marker_points(marker_points, self.best_matrix)
        print("\nTransformed marker data:")
        for marker in transformed_points:
            pt = marker["point"]
            print(f"Marker {marker['id']}: ({pt.x}, {pt.y}, {pt.z})")
        
        # 3. จัดเรียง marker ตาม id (ascending)
        sorted_markers = sorted(transformed_points, key=lambda m: m["id"])
        
        # 4. สำหรับแต่ละ marker สร้าง pose โดยใช้ค่า x, z จากผลลัพธ์ที่แปลงแล้ว
        #    แต่ fix ค่า y ด้วย self.FIX_Y แล้วรวมกับ Test_RPY (ค่า RPY คงที่)
        for marker in sorted_markers:
            marker_id = marker["id"]
            point = marker["point"]
            pose = [point.x, self.FIX_Y, point.z] + self.Test_RPY
            print(f"\nMoving to marker ID {marker_id} at pose: {pose}")
            
            # 5. สั่งหุ่นยนต์เคลื่อนที่ไปยัง pose ที่กำหนด
            self.robot.robot_moveL(pose, self.speed)
            time.sleep(3)
            
            # (ตัวอย่างนี้สามารถสั่ง gripper close/open หากต้องการจับ object ได้ที่นี่)
            
            # สั่งหุ่นยนต์กลับไปตำแหน่ง home หลังจากการเคลื่อนที่
            self.move_home()
            time.sleep(3)
            print(f"Completed move to marker ID {marker_id}")

def main():
    try:
        move2object = Move2Object()
        move2object.move_home()
        time.sleep(2)
        move2object.move_muti_to_object()
        time.sleep(5)
        move2object.move_home()
        move2object.stop_all()
    except Exception as e:
        print(f"Error initializing Move2Object: {e}")
        return

if __name__ == "__main__":
    main()
