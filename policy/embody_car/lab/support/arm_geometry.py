import math
MY_CONFIG = {
    # Label 1 (-90, 0, 90): angle_low=-90, angle_mid=0, angle_high=90
    "shoulder_pan":  {"id": 1, "range_min": 715,  "range_max": 3466, "a_low": 3088, "a_mid": 2005, "a_high": 963,  "label": 1},
    "wrist_flex":    {"id": 4, "range_min": 845,  "range_max": 3176, "a_low": 1088, "a_mid": 2049, "a_high": 3073, "label": 1},
    "wrist_roll":    {"id": 5, "range_min": 0,    "range_max": 4095, "a_low": 1000, "a_mid": 2048, "a_high": 3100, "label": 1},

    # Label 0 (0, 90, 180): angle_low=0, angle_mid=90, angle_high=180
    "shoulder_lift": {"id": 2, "range_min": 822,  "range_max": 3226, "a_low": 3114, "a_mid": 2072, "a_high": 1051, "label": 0},
    "elbow_flex":    {"id": 3, "range_min": 908,  "range_max": 3123, "a_low": 1061, "a_mid": 2061, "a_high": 3114, "label": 0},
    "gripper":       {"id": 6, "range_min": 800, "range_max": 2302, "a_low": 800, "a_mid": 1507, "a_high": 2302, "label": 0},
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
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1313,  "range_min": 800, "range_max": 2302},
}
JOINT_ID_MAP = {
    'shoulder_pan': 1,
    'shoulder_lift': 2,
    'elbow_flex': 3,
    'wrist_flex': 4,
    'wrist_roll': 5,
    'gripper': 6,
}
