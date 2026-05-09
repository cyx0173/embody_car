import serial
import json
import time

# ─── 配置区（请确保与你的 Demo 5 一致） ────────────────────────────
PORT     = "/dev/cu.usbmodem5AE60562991"
BAUDRATE = 1_000_000
INST_WRITE = 3

# ──────────────────────────────────────────────────────────

def checksum(packet_bytes):
    """完全套用 Demo 5 的校验逻辑"""
    return ~sum(packet_bytes) & 0xFF

def send_write(ser, servo_id, address, data_bytes):
    """完全套用 Demo 5 的发送逻辑"""
    params  = [address] + data_bytes
    length  = 2 + len(params)
    payload = [servo_id, length, INST_WRITE] + params
    packet  = [0xFF, 0xFF] + payload + [checksum(payload)]
    ser.write(bytes(packet))
    ser.read(64) # 清理回包

def move_to_joint_position(ser, servo_id, position, speed=500, acc=50):
    """
    仿照 Demo 5 的 spin 函数，但在关节模式下控制位置。
    地址 41 开始写 7 个字节：
    [加速度, 位置Lo, 位置Hi, 时间Lo, 时间Hi, 速度Lo, 速度Hi]
    """
    pos_lo = position & 0xFF
    pos_hi = (position >> 8) & 0xFF
    
    spd_lo = speed & 0xFF
    spd_hi = (speed >> 8) & 0xFF

    # 41:acc, 42-43:pos, 44-45:time(0), 46-47:speed
    send_write(ser, servo_id, 41, [acc, pos_lo, pos_hi, 0, 0, spd_lo, spd_hi])

# ──── 主程序 ────────────────────────────────────────────
try:
    # 1. 读取保存的姿态
    with open("sky_pose.json", "r") as f:
        saved_pose = json.load(f)
    print(f"成功读取姿态数据: {saved_pose}")

    with serial.Serial(PORT, BAUDRATE, timeout=0.5) as ser:
        print(f"连接串口 {PORT}...")

        # 遍历所有保存的电机 ID
        for mid_str, pos in saved_pose.items():
            mid = int(mid_str) # JSON 里的键是字符串，转成整数
            
            print(f"正在配置电机 ID={mid} ...")
            
            # A. 确保回到关节模式 (地址 33 写 0)
            send_write(ser, mid, 33, [0])
            time.sleep(0.05)
            
            # B. 开启扭矩/释放刹车 (地址 40 写 1)
            send_write(ser, mid, 40, [1])
            time.sleep(0.05)

            # C. 发送移动指令
            print(f" -> 移动到位置: {pos}")
            move_to_joint_position(ser, mid, pos, speed=400, acc=40)

        print("\n🚀 正在复原姿态，请观察机械臂动作...")
        time.sleep(3)
        print("复原完成！")

except FileNotFoundError:
    print("❌ 错误：找不到 sky_pose.json 文件！请先运行记录程序。")
except Exception as e:
    print(f"❌ 运行出错: {e}")