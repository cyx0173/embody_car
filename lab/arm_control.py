from turtle import pos
import serial
import time

class ServoController:
    REG_MODE = 33
    REG_WRITE_V = 41
    REG_POS_READ = 56

    ALL_IDS = [1, 2, 3, 4, 5, 6]
    HOME_POS = {1: 2185, 2: 863, 3: 3107, 4: 1584, 5: 77, 6: -1}

    def __init__(self, port="/dev/cu.usbmodem5AE60562991", baudrate=1_000_000):
        try:
            self._ser = serial.Serial(port, baudrate, timeout=0.01)
            self._states = {sid: {'mode': None, 'speed': None} for sid in self.ALL_IDS}
            print(f"✅ 串口已连接: {port}")
        except Exception as e:
            print(f"❌ 无法打开串口: {e}")
            raise

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

    def move_to(self, servo_id, pos, speed=1000, acc=50):
        self.set_mode(servo_id, 0)
        pos = max(0, min(4095, int(pos)))
        p_lo, p_hi = pos & 0xFF, (pos >> 8) & 0xFF
        s_lo, s_hi = int(speed) & 0xFF, (int(speed) >> 8) & 0xFF
        self._send_write(servo_id, self.REG_WRITE_V, [acc, p_lo, p_hi, 0, 0, s_lo, s_hi])
        self._states[servo_id]['speed'] = None

    def brake(self, servo_id):
        self.spin(servo_id, 0, acc=255)

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
if __name__ == "__main__":
    arm = ServoController()
    #arm.reset()
    pos5 = arm.get_position(5)
    print(f"5号电机位置: {pos5}")
    pos4 = arm.get_position(4)
    print(f"4号电机位置: {pos4}")
