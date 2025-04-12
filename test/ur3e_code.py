import cv2
import cv2.aruco as aruco
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface as RTDEIO
import time

class CameraHandler:
    def __init__(self, camera_index=0, frame_width=3840, frame_height=2160):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise Exception("ไม่สามารถเปิดกล้องได้")
        
        if self.cap.get(cv2.CAP_PROP_AUTOFOCUS) == -1:
            print("กล้องไม่รองรับการปรับโฟกัสอัตโนมัติ")
        else:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
    def wait_for_focus(self, wait_time=5000):
        cv2.waitKey(wait_time)
        
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("ไม่สามารถอ่านเฟรมจากกล้องได้")
        return frame
    
    def crop_and_zoom(self, frame, zoom_factor=1.3):
        
        height, width, _ = frame.shape
        start_row, start_col = int(height * ((1 - (1 / zoom_factor)) / 2)), int(width * ((1 - (1 / zoom_factor)) / 2))
        end_row, end_col = int(height * (1 - (1 - (1 / zoom_factor)) / 2)), int(width * (1 - (1 - (1 / zoom_factor)) / 2))
        cropped_frame = frame[start_row:end_row, start_col:end_col]
        zoomed_frame = cv2.resize(cropped_frame, (width, height))
        return zoomed_frame
    
    def save_image(self, frame, filename='zoomed_captured_image.jpg'):
        cv2.imwrite(filename, frame)
    
    def show_image(self, frame, window_name='zoomed_frame'):
        #cv2.imshow(window_name, frame)
        cv2.destroyAllWindows()
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class ArUcoProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.corners = None
        self.ids = None

    def find_aruco_corners(self):
        """ค้นหา ArUco markers และคืนค่ามุมของ markers"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters()
        self.corners, self.ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=parameters)

    def order_corners(self):
        """จัดเรียงมุมของ markers ตาม ID ที่ระบุ"""
        ordered_corners = [None] * 4
        if self.ids is not None:
            for i, marker_id in enumerate(self.ids.flatten()):
                if marker_id in [0, 1, 2, 3]:
                    ordered_corners[marker_id] = self.corners[i][0]
        return ordered_corners

    def get_transformed_image(self):
        """ปรับตัดกรอบภาพตามตำแหน่งของ markers"""
        ordered_corners = self.order_corners()
        if any(corner is None for corner in ordered_corners):
            raise ValueError("All marker IDs must be detected")

        src_corners = np.array([
            ordered_corners[0][0],  # มุมซ้ายบน (ID 0)
            ordered_corners[1][1],  # มุมขวาบน (ID 1)
            ordered_corners[2][2],  # มุมขวาล่าง (ID 2)
            ordered_corners[3][3]   # มุมซ้ายล่าง (ID 3)
        ], dtype="float32")

        src_corners_flatten = src_corners.reshape(-1, 2)
        min_x = np.min(src_corners_flatten[:, 0])
        max_x = np.max(src_corners_flatten[:, 0])
        min_y = np.min(src_corners_flatten[:, 1])
        max_y = np.max(src_corners_flatten[:, 1])

        width = int(max_x - min_x)
        height = int(max_y - min_y)

        dst_corners = np.array([
            [0, 0],  # มุมซ้ายบน
            [width - 1, 0],  # มุมขวาบน
            [width - 1, height - 1],  # มุมขวาล่าง
            [0, height - 1]  # มุมซ้ายล่าง
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(src_corners_flatten, dst_corners)
        warped_image = cv2.warpPerspective(self.image, matrix, (width, height))
        
        return warped_image

    def process_and_save_image(self, output_path):
        """ประมวลผลและบันทึกภาพผลลัพธ์"""
        self.find_aruco_corners()
        if self.ids is not None and len(self.ids) >= 4:
            result_image = self.get_transformed_image()
            cv2.imwrite(output_path, result_image)
            
            # แสดงขนาดของภาพ
            height, width = result_image.shape[:2]
            print(f"Size of the transformed image: {width} x {height} pixels")
            
            # คำนวณความกว้างและความยาวของ marker แต่ละ ID
            widths = []
            heights = []
            for i, corner in enumerate(self.order_corners()):
                if corner is not None:
                    marker_width = np.linalg.norm(corner[0] - corner[1])
                    marker_height = np.linalg.norm(corner[1] - corner[2])
                    widths.append(marker_width)
                    heights.append(marker_height)
                    #print(f"Marker ID {i} width: {marker_width:.2f} pixels, height: {marker_height:.2f} pixels")

            if widths and heights:
                avg_width = np.mean(widths)
                avg_height = np.mean(heights)
                #avg = (avg_width+avg_height)/2
                print(f"Average width of markers: {avg_width:.2f} pixels")
                print(f"Average height of markers: {avg_height:.2f} pixels")
            
            return result_image 
        else:
            print("Markers not detected or insufficient number of markers.")
            return None

class ImageProcessor:
    def __init__(self, image, Top_Right, Top_Left, Bottom_Right, Bottom_Left, Height):
        self.warped = image
        self.TR = Top_Right 
        self.TL = Top_Left
        self.BR = Bottom_Right 
        self.BL = Bottom_Left 
        self.Height = Height
        if self.warped is not None:
            self.height, self.width = self.warped.shape[:2]
        else:
            self.width = 0
            self.height = 0
        self.final_mask = None
        self.object_positions = []

    def find_colors(self):
        if self.warped is not None:
            self.hsv_warped = cv2.cvtColor(self.warped, cv2.COLOR_BGR2HSV)
            
            # ตั้งค่าขอบเขตสีสำหรับการตรวจจับ
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            lower_green = np.array([40, 100, 100])
            upper_green = np.array([90, 255, 255])

            lower_cyan = np.array([90, 100, 100])
            upper_cyan = np.array([140, 255, 255])

            lower_blue = np.array([90, 90, 10])
            upper_blue = np.array([130, 255, 255])

            lower_purple = np.array([130, 100, 10])
            upper_purple = np.array([160, 255, 255])

            # สร้างมาสก์สำหรับแต่ละสี
            mask_red1 = cv2.inRange(self.hsv_warped, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(self.hsv_warped, lower_red2, upper_red2)
            mask_yellow = cv2.inRange(self.hsv_warped, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(self.hsv_warped, lower_green, upper_green)
            mask_cyan = cv2.inRange(self.hsv_warped, lower_cyan, upper_cyan)
            mask_blue = cv2.inRange(self.hsv_warped, lower_blue, upper_blue)
            mask_purple = cv2.inRange(self.hsv_warped, lower_purple, upper_purple)

            self.final_mask = mask_red1 | mask_red2 | mask_yellow | mask_green | mask_cyan | mask_blue | mask_purple
        else:
            raise ValueError("No warped image found. Make sure to pass a valid image.")

    def apply_mask(self):
        if self.final_mask is not None:
            self.result = cv2.bitwise_and(self.warped, self.warped, mask=self.final_mask)
            cv2.imwrite('mask.jpg', self.result)
        else:
            raise ValueError("No final mask found. Run find_colors first.")

    def preprocessing(self):
        if self.result is not None:
            self.hsv_warped = cv2.cvtColor(self.result, cv2.COLOR_BGR2HSV)
            self.gray_warped = cv2.cvtColor(self.hsv_warped, cv2.COLOR_BGR2GRAY)
            _, self.binary_image = cv2.threshold(self.gray_warped, 100, 255, cv2.THRESH_BINARY_INV)
            self.inverted_image = cv2.bitwise_not(self.binary_image)
        else:
            raise ValueError("No result image found. Run apply_mask first.")

    def find_Contour(self):
        if self.inverted_image is not None:
            self.contours, _ = cv2.findContours(self.inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise ValueError("No inverted image found. Run preprocessing first.")

    def find_cube(self):
        if self.warped is not None and self.contours is not None:
            min_cx = float('inf')  # Initialize the minimum cx as infinity
            min_cy = float('inf')  # Initialize the minimum cy as infinity
            min_point = None  # Initialize the minimum point as None
            Hull_cx = float('inf')  # Initialize the minimum cx as infinity
            Hull_cy = float('inf')  # Initialize the minimum cy as infinity
            Hullpoint = None  # Initialize the minimum point as None

            for self.contour in self.contours:
                self.area = cv2.contourArea(self.contour)
                M = cv2.moments(self.contour)

                if self.area >= 1500 and self.area <= 15000:
                    cv2.drawContours(self.warped, [self.contour], -1, (255, 0, 0), 2)
                    #cv2.imwrite('contour.jpg', self.contours)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Extract values from A, B, C, D
                        TRx, TRy = self.TR
                        TLx, TLy = self.TL
                        BRx, BRy = self.BR
                        BLx, BLy = self.BL
                        Height = self.Height

                        lx = (TLx + (cx * ((TRx - TLx) / self.width))) / 1000
                        ly = (TLy - (cy * ((TLy - BLy) / self.height))) / 1000
                        lz = Height / 1000
                        self.object_positions.append((lx, ly, lz))

                        # Update the minimum cx and cy
                        if cx < min_cx:
                            min_cx = cx
                            min_cy = cy
                            min_point = (cx, cy)

                        cv2.circle(self.warped, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.putText(self.warped, f'({lx:.4f},{ly:.4f})', (cx - 50, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Approximate เส้นรอบ Contour เป็นรูปสี่เหลี่ยม
                    epsilon = 1 * cv2.arcLength(self.contour, True)
                    approx = cv2.approxPolyDP(self.contour, epsilon, True)
                    cv2.drawContours(self.warped, [approx], -1, (255, 0, 0), 2)
                    
                elif self.area > 5000:
                    hull = cv2.convexHull(self.contour)
                    cv2.drawContours(self.warped, [hull], -1, (0, 0, 0), 1)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(self.warped, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.putText(self.warped, f'({cx:.4f},{cy:.4f})', (cx - 50, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if cx < Hull_cx:
                            Hull_cx = cx
                            Hull_cy = cy
                            Hullpoint = (cx, cy)

                    else :
                        leftmost_point = tuple(hull[hull[:, :, 0].argmin()][0])
                        cx, cy = leftmost_point
                        
            if Hullpoint is not None :
                Hullpoint = Hullpoint
            else:
                Hullpoint = None
    
            # Calculate the adjusted position of the minimum point
            if min_point is not None :
                TRx, TRy = self.TR
                TLx, TLy = self.TL
                BRx, BRy = self.BR
                BLx, BLy = self.BL
                Height = self.Height

                if Hullpoint is not None :
                    if min_point[0] <= Hullpoint[0] :
                        x = (TLx + (min_point[0] * ((TRx - TLx) / self.width))) / 1000
                        y = (TLy - (min_point[1] * ((TLy - BLy) / self.height))) / 1000
                        z = self.Height/1000
                                    
                    else :
                        x = (TLx + (Hullpoint[0] * ((TRx - TLx) / self.width))) / 1000
                        y = (TLy - (Hullpoint[1] * ((TLy - BLy) / self.height))) / 1000
                        z = self.Height/1000
                                                
                else :
                    x = (TLx + (min_point[0] * ((TRx - TLx) / self.width))) / 1000
                    y = (TLy - (min_point[1] * ((TLy - BLy) / self.height))) / 1000
                    z = self.Height/1000
                              

                adjusted_min_point = (round(x, 3), round(y, 3) , round(z, 3))
                print(f"Minimum contour point: ({adjusted_min_point[0]}, {adjusted_min_point[1]}, {adjusted_min_point[2]})")
                return adjusted_min_point
            
            elif min_point is None and Hullpoint is not None:
                TRx, TRy = self.TR
                TLx, TLy = self.TL
                BRx, BRy = self.BR
                BLx, BLy = self.BL
                Height = self.Height
                x = (TLx + (Hullpoint[0] * ((TRx - TLx) / self.width))) / 1000
                y = (TLy - (Hullpoint[1] * ((TLy - BLy) / self.height))) / 1000
                z = self.Height/1000
                adjusted_min_point = (round(x, 3), round(y, 3) , round(z, 3))
                print(f"Minimum Hull point: ({adjusted_min_point[0]}, {adjusted_min_point[1]}, {adjusted_min_point[2]})")

                return adjusted_min_point
                
            else:
                print("No contour points found.")
                return None
           
        else:
            raise ValueError("No warped image or contours found. Run process_image methods in order.")

    def process_image(self):
        self.find_colors()
        self.apply_mask()
        self.preprocessing()
        self.find_Contour()
        self.find_cube()

    def show_result(self):
        if self.warped is not None and self.final_mask is not None:
            #cv2.imshow('Warped Image', self.warped)
            #cv2.imshow('Final Mask', self.final_mask)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No result to show. Run process_image first.")

class RobotController:
    def __init__(self, rtde_c, rtde_io , pick_coords):
        self.rtde_c = rtde_c
        self.rtde_io = rtde_io
        self.pick_coords = pick_coords

    def move(self, rtde_c, rtde_io , pick_coords):
        self.rtde_c = rtde_c
        self.rtde_io = rtde_io
        self.pick_coords = pick_coords
        self.velocity = 1
        self.acceleration = 1
        self.blend_1 = 0.0
        self.blend_2 = 0.02
        self.blend_3 = 0.0
        self.boolean = 0

        if self.pick_coords is not None and self.pick_coords[0] != 0 and self.pick_coords[1] != 0  :
            # Define paths
            self.path_Home = [0.31285, 0.09335, 0.30377, 1.537, -2.739, 0.002]
            self.path_Hide = [0.18253, 0.11303, 0.49689, 1.599, -2.724, -0.038]
            self.path_Pick = [self.pick_coords[0], self.pick_coords[1], self.pick_coords[2]+0.08, 1.599, -2.724, -0.038]
            self.path_Pick2 = [self.pick_coords[0], self.pick_coords[1], self.pick_coords[2], 1.599, -2.724, -0.038]
            self.path_Lift = [0.31285, 0.09325, 0.38041, 1.537, -2.739, 0.002]
            self.path_B41Place = [0.34423, 0.09186, 0.51680, 1.868, -3.294, 1.504]
            self.path_B42Place = [0.00306, 0.26659, 0.51680, 2.804, -2.978, 1.052]
            self.path_Place = [-0.11508, 0.23629, 0.53287, 4.001, -1.533, -0.756]
            self.path_Clear = [0.51371, 0.00817, 0.19568, 1.685, -3.116, 0.569]
            self.path_Clear1 = [0.41281, 0.01353, 0.21842, 1.428, -2.477, -0.169]
            self.path_Clear2 = [0.40417, -0.9367, 0.21983, 1.496, -2.411, -0.163]
            self.path_Clear3 = [0.49513, -0.09610, 0.21350, 1.933, -2.95, 0.429]
            self.path_Clear4 = [0.49842, 0.06843, 0.21361, 2.117, -2.874, 0.245]
            self.path_Clear5 = [0.40879, 0.06682, 0.21064, 1.694, -2.382, -0.343]

            
            # Execute movements
            self.rtde_c.moveJ_IK(self.path_Hide, self.velocity, self.acceleration)
            self.rtde_io.setStandardDigitalOut(2, True)
            self.rtde_io.setStandardDigitalOut(3, False)
            self.rtde_io.setStandardDigitalOut(1, True)
            self.rtde_io.setStandardDigitalOut(0, False)
            print(self.pick_coords[0] , self.pick_coords[1], self.pick_coords[2])


            if self.pick_coords[0] - 0.04 <= 0.235 :  # Too close.
                self.rtde_c.moveJ_IK(self.path_Pick, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Pick2, self.velocity, self.acceleration)
                self.boolean = 1
                self.rtde_io.setStandardDigitalOut(1, False)
                self.rtde_io.setStandardDigitalOut(0, True)
                self.rtde_io.setStandardDigitalOut(2, False)
                self.rtde_io.setStandardDigitalOut(3, True)
                time.sleep(1)

            elif self.pick_coords[0] - 0.1 >= 0.235 and self.pick_coords[0] <= 0.51 :  # normal
                self.path_Pick = [self.pick_coords[0] - 0.1 , self.pick_coords[1], self.pick_coords[2]+0.08, 1.599, -2.724, -0.038]
                self.path_Pick2 = [self.pick_coords[0] - 0.1 , self.pick_coords[1], self.pick_coords[2], 1.599, -2.724, -0.038]
                self.path_Pick3 = [self.pick_coords[0], self.pick_coords[1], self.pick_coords[2], 1.599, -2.724, -0.038]
                self.rtde_c.moveJ_IK(self.path_Pick, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Pick2, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Pick3, self.velocity, self.acceleration)
                self.boolean = 1
                self.rtde_io.setStandardDigitalOut(1, False)
                self.rtde_io.setStandardDigitalOut(0, True)
                self.rtde_io.setStandardDigitalOut(2, False)
                self.rtde_io.setStandardDigitalOut(3, True)
                time.sleep(1)

            elif self.pick_coords[0] - 0.04 >= 0.3 and self.pick_coords[0] <= 0.51 :  # normal
                self.path_Pick = [self.pick_coords[0] - 0.06 , self.pick_coords[1], self.pick_coords[2]+0.08, 1.599, -2.724, -0.038]
                self.path_Pick2 = [self.pick_coords[0] - 0.06 , self.pick_coords[1], self.pick_coords[2]+0.02, 1.599, -2.724, -0.038]
                self.path_Pick3 = [self.pick_coords[0], self.pick_coords[1], self.pick_coords[2], 1.599, -2.724, -0.038]
                self.rtde_c.moveJ_IK(self.path_Pick, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Pick2, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Pick3, self.velocity, self.acceleration)
                self.boolean = 1
                self.rtde_io.setStandardDigitalOut(1, False)
                self.rtde_io.setStandardDigitalOut(0, True)
                self.rtde_io.setStandardDigitalOut(2, False)
                self.rtde_io.setStandardDigitalOut(3, True)
                time.sleep(1)
    
            elif self.pick_coords[0] > 0.51  : # Too far
                print("Workpice Too far from Robot")
                for i in range(3) :
                    self.rtde_io.setStandardDigitalOut(3, True)
                    self.rtde_io.setStandardDigitalOut(2, False )
                    time.sleep(0.005)
                    self.rtde_io.setStandardDigitalOut(3, False)
                    self.rtde_io.setStandardDigitalOut(2, False)
            
    
            if self.boolean == 1: # back to Home
                
                self.rtde_io.setStandardDigitalOut(2, True)
                self.rtde_io.setStandardDigitalOut(3, False)

                self.rtde_c.moveJ_IK(self.path_B41Place, self.velocity, self.acceleration)
                self.rtde_c.moveJ_IK(self.path_Place, self.velocity, self.acceleration)

                self.rtde_io.setStandardDigitalOut(0, False)
                self.rtde_io.setStandardDigitalOut(1, True)
                self.rtde_io.setStandardDigitalOut(2, False)
                self.rtde_io.setStandardDigitalOut(3, True)
                time.sleep(0.5)

                self.rtde_io.setStandardDigitalOut(2, True)
                self.rtde_io.setStandardDigitalOut(3, False)
                self.rtde_c.moveJ_IK(self.path_B41Place, self.velocity, self.acceleration)

                self.rtde_c.moveJ_IK(self.path_Hide, self.velocity, self.acceleration)
                #self.rtde_io.setStandardDigitalOut(2, True)
                #self.rtde_io.setStandardDigitalOut(3, True)

            # Stop the script
            # self.rtde_c.stopScript()

        elif self.pick_coords is not None and self.pick_coords[0] == 900 and self.pick_coords[1] == 900  :
            self.rtde_io.setStandardDigitalOut(0, True)
            self.rtde_io.setStandardDigitalOut(1, False )
            self.rtde_io.setStandardDigitalOut(3, True)
            self.rtde_io.setStandardDigitalOut(2, False )
            self.rtde_c.moveJ_IK(self.path_Clear, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Clear1, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Clear2, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Clear3, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Clear4, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Clear5, self.velocity, self.acceleration)
            self.rtde_c.moveJ_IK(self.path_Hide, self.velocity, self.acceleration)
            self.rtde_io.setStandardDigitalOut(2, True)
            self.rtde_io.setStandardDigitalOut(3, False )
            self.rtde_io.setStandardDigitalOut(1, True)
            self.rtde_io.setStandardDigitalOut(0, False )

        else :
                print("Workpice not Detected")
                self.rtde_io.setStandardDigitalOut(3, True)
                self.rtde_io.setStandardDigitalOut(2, False )

                
    def Hide(self):
        self.rtde_c = rtde_c
        self.rtde_io = rtde_io
        self.pick_coords = pick_coords
        self.velocity = 0.5
        self.acceleration = 0.5
        self.blend_1 = 0.0
        self.blend_2 = 0.02
        self.blend_3 = 0.0
        self.boolean = 0
        self.path_Hide = [0.18253, 0.11303, 0.49689, 1.599, -2.724, -0.038]
        self.rtde_c.moveJ_IK(self.path_Hide, self.velocity, self.acceleration)
        self.rtde_io.setStandardDigitalOut(2, True)
        self.rtde_io.setStandardDigitalOut(0, True)

if __name__ == "__main__":
    ip_address = "172.16.4.11"
    rtde_c = RTDEControl(ip_address)
    rtde_io = RTDEIO(ip_address)
    # SET Edge Position  [X,Y] (mm)
    Top_Right    = [622.76, 137.63]        
    Top_Left     = [182.76, 137.63]
    Bottom_Right = [623.28, -134.52]
    Bottom_Left  = [183.28, -134.52]
    Height = 220  # (mm)  Height of gripper when picking up
    pick_coords = (0.18253, 0.11303, 0.49689) 
    robot = RobotController(rtde_c, rtde_io , pick_coords)
    robot.Hide()

    

    while (True):
        camera_handler = CameraHandler(1)
        # get position
        # ------------block space alert
        # set min X  and max <= 0.5
        # Stop QR
        # [self.pick_coords[0], self.pick_coords[1], self.pick_coords[2]+0.08, 1.599, -2.724, -0.038]

        try:
            camera_handler.wait_for_focus(10)  ## 0
            frame = camera_handler.capture_frame()
            zoomed_frame = camera_handler.crop_and_zoom(frame, 1.3)
            camera_handler.show_image(zoomed_frame)
            camera_handler.save_image(zoomed_frame)
            print("บันทึกภาพซูมสำเร็จ")
            cv2.waitKey(0)  # Wait for a key press to close the image window
        except Exception as e:
            print(e)
        finally:
            camera_handler.release()

        image_path = 'zoomed_captured_image.jpg'
        output_path = 'transformed_aruco_image.jpg'
        aruco_processor = ArUcoProcessor(image_path)
        transformed_image = aruco_processor.process_and_save_image(output_path)

        if transformed_image is not None:
            image_processor = ImageProcessor(transformed_image, Top_Right, Top_Left, Bottom_Right, Bottom_Left, Height)
            image_processor.process_image()
            image_processor.show_result()
            adjusted_min_point = image_processor.find_cube()
            pick_coords = adjusted_min_point
            robot = RobotController(rtde_c, rtde_io , pick_coords)
            robot.move(rtde_c, rtde_io , pick_coords)

        else:
            print("Failed to get transformed image from ArUcoProcessor")
