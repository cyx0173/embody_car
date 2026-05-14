import math
import numpy as np
class Arm_Solver:
    def __init__(self):
        self.arm = Arm_position()
    def calculate_arm_position(self,sita1,sita2,sita3,sita4,sita5):
        r1,z1 = self.arm.calculate_arm1(sita1)
        r2,z2 = self.arm.calculate_arm2(sita2)
        r3,z3 = self.arm.calculate_arm3(sita3)
        r4,z4 = self.arm.calculate_arm4(sita4)  
        R = r1 + r2 + r3 + r4 
        Z = z1 + z2 + z3 + z4 
        return R,Z
class Arm_position:
    def __init__(self):
        self.R = [
            0.03, 
            0.12, 
            0.13, 
            0.1
        ]
    def sita_convert(self,sita):

        return sita * math.pi / 180
    def calculate_arm1(self,sita,r_base):
        r = -0.03
        z = 0.07 
        return r,z 
    def calculate_arm2(self,sita):
        r = self.R[1] * math.cos(sita)
        z = self.R[1] * math.sin(sita)
        return r,z 
    def calculate_arm3(self,sita):
        r = self.R[2] * math.cos(sita)
        z = self.R[2] * math.sin(sita)
        return r,z 
    def calculate_arm4(self,sita):
        r = self.R[3] * math.cos(sita)
        z = self.R[3] * math.sin(sita)
        return r,z 
if __name__ == "__main__":
    arm_position = Arm_position()
    r,z = arm_position.calculate_arm2(0)
    print(r,z)
