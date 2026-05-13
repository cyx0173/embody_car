import math
MY_CONFIG = {
    # Label 1 (-90, 0, 90): angle_low=-90, angle_mid=0, angle_high=90
    "shoulder_pan":  {"id": 1, "range_min": 715,  "range_max": 3466, "a_low": 3088, "a_mid": 2005, "a_high": 963,  "label": 1},
    "wrist_flex":    {"id": 4, "range_min": 845,  "range_max": 3176, "a_low": 1088, "a_mid": 2049, "a_high": 3073, "label": 1},
    "wrist_roll":    {"id": 5, "range_min": 0,    "range_max": 4095, "a_low": 1000, "a_mid": 2048, "a_high": 3100, "label": 1},

    # Label 0 (0, 90, 180): angle_low=0, angle_mid=90, angle_high=180
    "shoulder_lift": {"id": 2, "range_min": 822,  "range_max": 3226, "a_low": 3114, "a_mid": 2072, "a_high": 1051, "label": 0},
    "elbow_flex":    {"id": 3, "range_min": 908,  "range_max": 3123, "a_low": 1061, "a_mid": 2061, "a_high": 3114, "label": 0},
    "gripper":       {"id": 6, "range_min": 1507, "range_max": 3026, "a_low": 1507, "a_mid": 2266, "a_high": 3026, "label": 0},
}
def angles_to_ticks(joints_rad: dict[str, float]) -> dict[int, int]:
    commands = {}
    config = MY_CONFIG
    for name, rad in joints_rad.items():
        if name not in config: continue
        c = config[name]
        deg = math.degrees(rad)
        target = deg if c["label"] == 0 else deg + 0 
        ref_low, ref_mid, ref_high = (0, 90, 180) if c["label"] == 0 else (-90, 0, 90)
        p_low, p_mid, p_high = c["a_low"], c["a_mid"], c["a_high"]
        if deg <= ref_mid:
            pct = (deg - ref_low) / (ref_mid - ref_low)
            raw = p_low + (p_mid - p_low) * pct
        else:
            pct = (deg - ref_mid) / (ref_high - ref_mid)
            raw = p_mid + (p_high - p_mid) * pct
        final_tick = max(c["range_min"], min(c["range_max"], int(round(raw))))
        commands[c["id"]] = final_tick
    return commands

SERVO_CALIBRATION = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 2005,  "range_min": 715,  "range_max": 3466},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 3114, "range_min": 822,  "range_max": 3226},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1061,  "range_min": 908,  "range_max": 3123},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": 2049,  "range_min": 845,  "range_max": 3176},
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
            1: AxisConfig(left=2510, center=4084, right=1161,min=715,max=3466,angel_0=2024,angel_180=3098),
            2: AxisConfig(left=822, center=3210, right=4021,min=822,max=3226,angel_0=3217,angel_180=3098),
            3: AxisConfig(left=2601, center=4061, right=733,min=908,max=3123,angel_0=1289,angel_180=3098),
            4: AxisConfig(left=2194, center=4046, right=444,min=845,max=3176,angel_0=1904,angel_180=3098),
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
(embody) chengyx@chengyxdeMacBook-Air lab % python arm_control.py
✅ 串口已连接: /dev/cu.usbmodem5AE60562991
舵机 ID 1 的当前原始脉冲 (Raw Ticks): 2005
舵机 ID 3 的当前原始脉冲 (Raw Ticks): 1061
舵机 ID 4 的当前原始脉冲 (Raw Ticks): 2049
(embody) chengyx@chengyxdeMacBook-Air lab % 
这是初始的0度下的 span(1,+300)从+90到-60
舵机 ID 1 的当前原始脉冲 (Raw Ticks): 2005 0度 
舵机 ID 1 的当前原始脉冲 (Raw Ticks): 963  +90度 （左侧）
舵机 ID 1 的当前原始脉冲 (Raw Ticks): 3088 -90度 （右侧）


舵机 ID 2 的当前原始脉冲 (Raw Ticks): 3114 0度
舵机 ID 2 的当前原始脉冲 (Raw Ticks): 2072 +90度
舵机 ID 2 的当前原始脉冲 (Raw Ticks): 1051 +180度 

舵机 ID 3 的当前原始脉冲 (Raw Ticks): 1061 0度
舵机 ID 3 的当前原始脉冲 (Raw Ticks): 2061 +90度
舵机 ID 3 的当前原始脉冲 (Raw Ticks): 3114 +180度

舵机 ID 4 的当前原始脉冲 (Raw Ticks): 2049 0度
舵机 ID 4 的当前原始脉冲 (Raw Ticks): 3073 +90度 (朝下)
舵机 ID 4 的当前原始脉冲 (Raw Ticks): 1088 -90度 (朝上)
span(4,-300)从朝下到朝上
'''