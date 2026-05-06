      
#!/usr/bin/env python
"""
Demo 5: 轮子模式（连续转动，多圈）
=====================================

【普通关节模式 vs 轮子模式的区别】

  关节模式（默认）：舵机有固定的起点和终点（0~4095），
                   你告诉它"转到第 2000 格"，它就停在那里不动。

  轮子模式：        取消位置限制，舵机像电机一样持续旋转。
                   你告诉它"以这个速度转"，它就一直转，直到你让它停。

【切换方式】
  往地址 33（STS_MODE）写 1 → 进入轮子模式
  往地址 33            写 0 → 回到关节模式（默认）

  注意：STS_MODE 在 SRAM 里（不是 EPROM），所以不需要解锁，
        但断电后会恢复默认值（关节模式），每次上电都要重新设置。

【速度控制】
  轮子模式下，仍然往地址 41 写 7 个字节，但位置字段留 0，只填速度。
  速度字段（地址 46~47，2字节）的编码规则：
    - 最高位（bit 15）是方向位：0 = 正转，1 = 反转
    - 其余 15 位是速度大小，范围 0~3400（0 = 停止）

  举例：
    速度 +500（正转）→ 0b 0000_0001_1111_0100 = 0x01F4
    速度 -500（反转）→ 0b 1000_0001_1111_0100 = 0x81F4
"""

import serial   # pip install pyserial
import time

# ─── 配置区（根据实际情况修改）────────────────────────────
PORT     = "/dev/tty.usbmodem5AB01816651"
BAUDRATE = 1_000_000
SERVO_ID = 7
# ──────────────────────────────────────────────────────────

INST_WRITE = 3


def checksum(packet_bytes):
    return ~sum(packet_bytes) & 0xFF


def send_write(ser, servo_id, address, data_bytes):
    params  = [address] + data_bytes
    length  = 2 + len(params)
    payload = [servo_id, length, INST_WRITE] + params
    packet  = [0xFF, 0xFF] + payload + [checksum(payload)]
    ser.write(bytes(packet))
    ser.read(64)


def set_wheel_mode(ser, servo_id):
    """切换到轮子模式：往地址 33 写 1"""
    send_write(ser, servo_id, 33, [1])


def set_joint_mode(ser, servo_id):
    """切换回关节模式：往地址 33 写 0"""
    send_write(ser, servo_id, 33, [0])


def spin(ser, servo_id, speed, acc=50):
    """
    设定轮子转速。

    speed : 正数 = 正转，负数 = 反转，0 = 停止。范围 -3400 ~ +3400。
    acc   : 加速度，1~254，越小启动越平缓。

    速度编码：负数时把绝对值的最高位（bit 15）置 1。
    """
    if speed < 0:
        raw_speed = (-speed) | 0x8000   # 设置方向位
    else:
        raw_speed = speed

    spd_lo = raw_speed & 0xFF
    spd_hi = (raw_speed >> 8) & 0xFF

    # 从地址 41 写 7 字节：acc, pos=0, pos=0, time=0, time=0, spd_lo, spd_hi
    send_write(ser, servo_id, 41, [acc, 0, 0, 0, 0, spd_lo, spd_hi])


# ──── 主程序 ────────────────────────────────────────────
print(f"连接串口 {PORT}，控制舵机 ID={SERVO_ID}")
print()

with serial.Serial(PORT, BAUDRATE, timeout=0.5) as ser:

    print("切换到轮子模式 ...")
    set_wheel_mode(ser, SERVO_ID)
    time.sleep(0.1)

    print("正转，速度 1000，持续 2 秒 ...")
    spin(ser, SERVO_ID, speed=1000, acc=50)
    time.sleep(2)

    print("停止，0.5 秒 ...")
    spin(ser, SERVO_ID, speed=0)
    time.sleep(0.5)

    print("反转，速度 1000，持续 2 秒 ...")
    spin(ser, SERVO_ID, speed=-1000, acc=50)
    time.sleep(2)

    print("加速到 2400，持续 1 秒 ...")
    spin(ser, SERVO_ID, speed=2400, acc=100)
    time.sleep(1)

    print("缓慢减速停止 ...")
    spin(ser, SERVO_ID, speed=0, acc=20)   # acc 小 = 减速更缓
    time.sleep(1)

    print("切回关节模式（断电也会自动恢复，这里只是演示）")
    set_joint_mode(ser, SERVO_ID)

print("完成！")

    