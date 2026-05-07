class AxisConfig:
    def __init__(self, left, center, right):
        self.l = left
        self.c = center
        self.r = right

class ArmManager:
    def __init__(self):
        self.axes = {
            1: AxisConfig(left=2510, center=4084, right=1161),
            2: AxisConfig(left=3256, center=3210, right=4021),
            3: AxisConfig(left=2601, center=4061, right=733),
            4: AxisConfig(left=2194, center=4046, right=444),
            5: AxisConfig(left=1961, center=4093, right=1756),
            6: AxisConfig(left=2710, center=3210, right=1500),
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
