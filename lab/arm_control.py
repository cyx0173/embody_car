from turtle import pos
import serial
import time
from Angle_config import SERVO_CALIBRATION, JOINT_ID_MAP

def _resolve_zero_position(joint_config: dict) -> int:
    lower = int(joint_config["range_min"])
    upper = int(joint_config["range_max"])
    offset = int(joint_config["homing_offset"])
    candidates = (offset, (offset % 4096), 2048 + offset, 2048 - offset)
    for c in candidates:
        if lower <= int(c) <= upper:
            return int(c)
    return int((lower + upper) / 2)


class ServoController:
    REG_MODE = 33
    REG_WRITE_V = 41
    REG_POS_READ = 56

    ALL_IDS = [1, 2, 3, 4, 5, 6]
    WHEEL_IDS = [7, 8, 9, 10]
    HOME_POS = {1: 2185, 2: 863, 3: 3107, 4: 1245, 5: 312}

    def __init__(self, port="/dev/cu.usbmodem5AE60562991", baudrate=1_000_000):
        try:
            self._ser = serial.Serial(port, baudrate, timeout=0.01)
            self._states = {sid: {'mode': None, 'speed': None} for sid in self.ALL_IDS}
            print(f"✅ 串口已连接: {port}")
        except Exception as e:
            print(f"❌ 无法打开串口: {e}")
            raise
        self.middle = {
            name: _resolve_zero_position(cfg) for name, cfg in SERVO_CALIBRATION.items()
        }
        self.joints_range = {
            name: [cfg["range_min"], cfg["range_max"]] for name, cfg in SERVO_CALIBRATION.items()
        }
        self._data = {
            "shoulder_pan":   {"id": 1, "drive_mode": 0, "homing_offset": 2005,  "range_min": 715,  "range_max": 3466},
            "shoulder_lift":  {"id": 2, "drive_mode": 0, "homing_offset": 3114, "range_min": 822,  "range_max": 3226},
            "elbow_flex":     {"id": 3, "drive_mode": 0, "homing_offset": 1061,  "range_min": 908,  "range_max": 3123},
            "wrist_flex":     {"id": 4, "drive_mode": 0, "homing_offset": 2049,  "range_min": 845,  "range_max": 3176},
            "wrist_roll":     {"id": 5, "drive_mode": 0, "homing_offset": 1900,  "range_min": 0,    "range_max": 4095},
            "gripper":        {"id": 6, "drive_mode": 0, "homing_offset": 1313,  "range_min": 1507, "range_max": 3026},
        }


    def joints_move_angle(self, joints_name_position: list, speed: int = 1000, acc: int = 50):
        for name, angle_deg in joints_name_position:
            servo_id = JOINT_ID_MAP[name]
            raw = int(angle_deg * 4096.0 / 360.0 + self.middle[name])
            lo, hi = self.joints_range[name]
            raw = max(lo, min(hi, raw))
            self.move_to(servo_id, raw, speed=speed, acc=acc)

    def _checksum(self, payload):
        return (~sum(payload)) & 0xFF

    def _send_write(self, servo_id, address, data):
        self._ser.reset_input_buffer()
        params = [address] + data
        length = 2 + len(params)
        payload = [servo_id, length, 3] + params
        packet = [0xFF, 0xFF] + payload + [self._checksum(payload)]
        self._ser.write(bytes(packet))
        self._ser.flush()

    def _send_read(self, servo_id, address, read_len=2):
        self._ser.reset_input_buffer()
        params = [address, read_len]
        length = 2 + len(params)
        payload = [servo_id, length, 2] + params
        packet = [0xFF, 0xFF] + payload + [self._checksum(payload)]
        self._ser.write(bytes(packet))

        time.sleep(0.015)
        response = self._ser.read(read_len + 6)
        if (len(response) >= read_len + 6
                and response[0] == 0xFF
                and response[1] == 0xFF
                and (~sum(response[2:-1]) & 0xFF) == response[-1]):
            return int.from_bytes(response[5:-1], byteorder='little')
        return -1

    def set_mode(self, servo_id, mode):
        if servo_id not in self._states:
            self._states[servo_id] = {'mode': None, 'speed': None}
        if self._states[servo_id]['mode'] != mode:
            self._send_write(servo_id, self.REG_MODE, [mode])
            self._states[servo_id]['mode'] = mode
            time.sleep(0.005)

    def spin(self, servo_id, speed, acc=50):
        self.set_mode(servo_id, 1)
        if self._states[servo_id]['speed'] == speed:
            return
        abs_speed = abs(int(speed))
        raw_speed = (abs_speed | 0x8000) if speed < 0 else abs_speed
        spd_lo, spd_hi = raw_speed & 0xFF, (raw_speed >> 8) & 0xFF
        self._send_write(servo_id, self.REG_WRITE_V, [acc, 0, 0, 0, 0, spd_lo, spd_hi])
        self._states[servo_id]['speed'] = speed

    def move_to(self, servo_id, pos, speed=4000, acc=50):
        self.set_mode(servo_id, 0)
        pos = max(0, min(4095, int(pos)))
        p_lo, p_hi = pos & 0xFF, (pos >> 8) & 0xFF
        s_lo, s_hi = int(speed) & 0xFF, (int(speed) >> 8) & 0xFF
        self._send_write(servo_id, self.REG_WRITE_V, [acc, p_lo, p_hi, 0, 0, s_lo, s_hi])
        self._states[servo_id]['speed'] = None

    def brake(self, servo_id):
        self.spin(servo_id, 0, acc=255)

    def spin_wheel(self, speed = 100, acc=50):
        for i in (7, 10):
            self.spin(i, speed, acc=acc)
    
    def move_wheel(self, mode, speed, acc=50):
        speed = int(speed)
        half_speed = int(round(speed * 0.5))
        if mode == 0:
            self.spin(8, -speed, acc=acc)
            self.spin(9,  speed, acc=acc)
            self.spin(7,  0,     acc=acc)
        elif mode == 1:
            self.spin(8,  speed, acc=acc)
            self.spin(9, -speed, acc=acc)
            self.spin(7,  0,     acc=acc)
        elif mode == 2:
            self.spin(8,  half_speed, acc=acc)
            self.spin(9,  half_speed, acc=acc)
            self.spin(7, -speed,      acc=acc)
        elif mode == 3:
            self.spin(8, -half_speed, acc=acc)
            self.spin(9, -half_speed, acc=acc)
            self.spin(7,  speed,      acc=acc)
        elif mode == 4:
            self.spin(8, 0, acc=255)
            self.spin(9, 0, acc=255)
            self.spin(7, 0, acc=255)
        else:
            raise ValueError(f"不支持的 mode: {mode}")

    def brake_wheels(self):
        self.move_wheel(4, 0, acc=255)
        for sid in self.WHEEL_IDS:
            self.brake(sid)

    def brake_all(self):
        for sid in self.ALL_IDS:
            self.brake(sid)

    def reset(self, speed=1000):
        print("🏠 正在全轴复位...")
        for sid, pos in self.HOME_POS.items():
            self.move_to(sid, pos, speed=speed)
        time.sleep(2)
    
    def get_position(self, servo_id):
        return self._send_read(servo_id, self.REG_POS_READ, 2)

    def close(self):
        self.brake_all()
        self._ser.close()

    def is_safe(self, servo_id, margin=20):
        raw = self.get_position(servo_id)
        print(raw)
        for cfg in self._data.values():
            if cfg["id"] == servo_id:
                lo = cfg["range_min"]
                hi = cfg["range_max"]
                return lo + margin <= raw <= hi - margin
        return False
    
if __name__ == "__main__":
    arm = ServoController()
    arm.reset()


'''
    arm.move_to(1,3022)
    time.sleep(0.5)
    arm.move_to(2,3069)
    time.sleep(0.5)
    arm.move_to(3,1024)
    time.sleep(0.5)
    arm.move_to(4,2031)
    time.sleep(0.5)
'''
'''
(embody) chengyx@chengyxdeMacBook-Air lab % python arm_control.py
✅ 串口已连接: /dev/cu.usbmodem5AE60562991
舵机 ID 1 的当前原始脉冲 (Raw Ticks): 2005
舵机 ID 2 的当前原始脉冲 (Raw Ticks): 3114
舵机 ID 3 的当前原始脉冲 (Raw Ticks): 1061
舵机 ID 4 的当前原始脉冲 (Raw Ticks): 2049
(embody) chengyx@chengyxdeMacBook-Air lab % 
这是初始的0度下的

舵机 ID 1 的当前原始脉冲 (Raw Ticks): 963  +90度 
舵机 ID 2 的当前原始脉冲 (Raw Ticks): 2072 +90度
舵机 ID 2 的当前原始脉冲 (Raw Ticks): 1051 +180度 

舵机 ID 3 的当前原始脉冲 (Raw Ticks): 2061 +90度
舵机 ID 3 的当前原始脉冲 (Raw Ticks): 3114 +180度

舵机 ID 4 的当前原始脉冲 (Raw Ticks): 3073 +90度 
舵机 ID 4 的当前原始脉冲 (Raw Ticks): 1088 -90度

'''
