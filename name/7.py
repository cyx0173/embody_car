      
#!/usr/bin/env python
"""
Demo 1: 设置舵机编号（ID）
===============================
飞特舵机的 ID 存储在舵机内部的 EPROM 中。
EPROM 默认是加锁的，修改前必须先解锁，改完后再重新上锁。

通信协议格式（每条指令都是这样一个字节串）：
  [0xFF] [0xFF] [ID] [Length] [Instruction] [Param0 ...] [Checksum]
   固定头  固定头  目标  后续字节数   指令类型      参数...        校验和

校验和算法：
  checksum = ~(ID + Length + Instruction + Param0 + ... + ParamN) & 0xFF
"""

import serial  # pip install pyserial

# ─── 配置区（根据实际情况修改）────────────────────────────
PORT      = " /dev/cu.usbmodem5AE60562991"   # 串口号，Mac 示例
BAUDRATE  = 1_000_000                  # 飞特舵机默认波特率 1 Mbps
OLD_ID    = 1                          # 舵机当前 ID
NEW_ID    = 7                         # 想要改成的新 ID
# ──────────────────────────────────────────────────────────


def checksum(packet_bytes):
    """校验和 = 对 ID、Length、Instruction、参数全部求和，取反，保留低 8 位"""
    return ~sum(packet_bytes) & 0xFF


def send_packet(ser, servo_id, instruction, params):
    """
    组装并发送一个完整的指令包，然后等待舵机回包。
    返回舵机回包的原始字节（调试用）。
    """
    length  = 2 + len(params)          # Length = 指令字节(1) + 参数字节数 + 校验和字节(1)
    payload = [servo_id, length, instruction] + params
    packet  = [0xFF, 0xFF] + payload + [checksum(payload)]

    ser.write(bytes(packet))
    response = ser.read(64)            # 最多读 64 字节的回包
    return response


# ──── 内存地址常量（来自 sts.py）─────────────────────────
ADDR_LOCK = 55   # EPROM 写保护锁（写 0 解锁，写 1 上锁）
ADDR_ID   = 5    # 舵机 ID 存储地址
# ──────────────────────────────────────────────────────────

INST_WRITE = 3   # 写指令

print(f"准备将舵机 ID 从 {OLD_ID} 改为 {NEW_ID}")

with serial.Serial(PORT, BAUDRATE, timeout=0.5) as ser:

    # 第一步：解锁 EPROM（写 0 到地址 55）
    print("Step 1: 解锁 EPROM ...")
    send_packet(ser, OLD_ID, INST_WRITE, [ADDR_LOCK, 0])

    # 第二步：把新 ID 写入地址 5
    print(f"Step 2: 写入新 ID = {NEW_ID} ...")
    send_packet(ser, OLD_ID, INST_WRITE, [ADDR_ID, NEW_ID])

    # 第三步：用新 ID 重新上锁 EPROM（写 1 到地址 55）
    print("Step 3: 重新锁定 EPROM ...")
    send_packet(ser, NEW_ID, INST_WRITE, [ADDR_LOCK, 1])

print(f"完成！舵机 ID 已更改为 {NEW_ID}")
print("提示：断电再上电后，新 ID 即可生效（部分型号立即生效）。")

    