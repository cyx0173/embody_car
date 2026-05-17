import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_control import ServoController
from Angle_config import ArmManager
import time 
arm = ServoController()
arm_manager = ArmManager()
arm.spin(1, 400)
while True:
    safe,danger = arm_manager.safe_detect(1,arm)
    pos = arm.get_position(1)
    print(pos)
    if not safe:
        arm.spin(1, -400)
    time.sleep(1)
