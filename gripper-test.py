from classrobot import gripper
import time

if __name__ == "__main__":
    gr = gripper.MyGripper3Finger()
    # Initialize the gripper
    gr.my_init(host="192.168.200.11", port=502)
    time.sleep(0.6)
    # Test closing
    gr.my_hand_close()
    time.sleep(2)
    # Test opening
    gr.my_hand_open()
    time.sleep(2)