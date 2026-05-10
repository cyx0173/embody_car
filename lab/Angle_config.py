SERVO_CALIBRATION = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1788,  "range_min": 715,  "range_max": 3466},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1706, "range_min": 822,  "range_max": 3226},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1712,  "range_min": 908,  "range_max": 3123},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": 1345,  "range_min": 845,  "range_max": 3176},
    "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": 1900,  "range_min": 0,    "range_max": 4095},
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1313,  "range_min": 1507, "range_max": 3026},
}
JOINT_ID_MAP = {
    'shoulder_pan': 1,
    'shoulder_lift': 2,
    'elbow_flex': 3,
    'wrist_flex': 4,
    'wrist_roll': 5,
    'gripper': 6,
}
class AxisConfig:
    def __init__(self, left, center, right,min,max,angel_0,angel_180):
        self.l = left
        self.c = center
        self.r = right
        self.min = min
        self.max = max
        self.angel_0 = angel_0
        self.angel_180 = angel_180

class ArmManager:
    def __init__(self):
        self.axes = {
            1: AxisConfig(left=2510, center=4084, right=1161,min=715,max=3466,angel_0=1062,angel_180=3098),
            2: AxisConfig(left=822, center=3210, right=4021,min=822,max=3226,angel_0=1062,angel_180=3098),
            3: AxisConfig(left=2601, center=4061, right=733,min=908,max=3123,angel_0=1062,angel_180=3098),
            4: AxisConfig(left=2194, center=4046, right=444,min=845,max=3176,angel_0=1062,angel_180=3098),
            5: AxisConfig(left=1961, center=4093, right=1756,min=0,max=4095,angel_0=1062,angel_180=3098),
            6: AxisConfig(left=2710, center=3210, right=1500,min=1507,max=3026,angel_0=1062,angel_180=3098),
        }

    def safe_detect(self, axis_id, arm, margin=20):
        raw = arm.get_position(axis_id)
        ax = self.axes[axis_id]
        safe_right = (raw >= ax.c) or (raw <= ax.r - margin)
        safe_left = (raw >= ax.l + margin) and (raw <= ax.c)
        if raw >= ax.l:
           raw -= ax.c + 200
        safe = safe_right or safe_left
        if not safe:
            if raw >= 0 : 
                danger = "right"
            elif raw <= 0:
                danger = "left"
        else:
            danger = None
        return safe,danger

    def get_direction_logic(self, axis_id, raw):
        ax = self.axes[axis_id]
        if (raw >= ax.c) or (raw <= ax.r):
            return 1
        if (raw >= ax.l) and (raw <= ax.c):
            return -1
        return 0
'''
    def angel_convert(self,axis_id,raw):
        ax = self.axes[axis_id]
        single_angel = abs(ax.angel_180 - ax.angel_0) / 180 
        if (raw >= ax.right)
'''