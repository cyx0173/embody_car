# 橘子抓取数据集录制与训练指南

这份文档用于教队友如何使用当前代码录制橘子抓取数据集，并训练 ACT 抓取模型。

日常只需要用两个脚本：

- `lab/record_orange_dataset.py`：用主臂、从臂和 wrist 相机录制 LeRobot 数据集。
- `lab/train_orange_act.py`：用录好的数据集训练 ACT policy，并实时显示训练日志。

当前默认目标物体是：

```text
orange
```

当前默认数据集是：

```text
datasets/orange_wrist_grasp_formal_clean_v1
repo_id=embody_car/orange_wrist_grasp_formal_clean_v1
```

## 1. 录制前准备

进入项目和环境：

```bash
cd /Users/kismet/embody_car
conda activate embody_car
```

硬件连接：

- 从臂 follower 驱动板：`/dev/cu.usbmodem5AE60562991`
- 主臂 leader 驱动板：`/dev/cu.usbmodem5AE60825831`
- wrist 相机：OpenCV index `1`

一次录制数据集时，尽量保持桌面、机械臂、相机位置不变。每条 episode 开始前，把橘子放到 wrist 相机视野里。

先做两个快速检查：

```bash
python lab/leader_follower_teleop.py --print-positions
python lab/record_orange_dataset.py --help
```

如果主臂和从臂跟随关系不对，或者有舵机读数跳变，不要开始录数据。先把 teleop 修好，否则录出来的数据会污染训练。

## 2. 先录 3 条试验数据

先录一个小 pilot 数据集：

```bash
python lab/record_orange_dataset.py \
  --num-episodes 3 \
  --overwrite \
  --show
```

每一条 episode 的流程是：

1. 从臂自动回到 reset-before 姿态，夹爪张开。
2. 程序等待你按 Enter。
3. 把橘子放到 wrist 相机画面里。
4. 手握主臂，准备示教。
5. 按 Enter 开始录制。
6. 用主臂控制从臂完成抓取动作。
7. 程序会记录 reset-after 动作，此时夹爪保持闭合。
8. 进入下一条 episode，再次等待 Enter。

如果当前 episode 动作已经完成，可以在预览窗口里按 `q` 提前结束当前 episode，并保存已经录到的帧。

一条好数据应该满足：

- 橘子在抓取前清楚出现在 wrist 相机画面里。
- 夹爪接近橘子的动作平滑。
- 夹爪确实闭合到橘子上。
- 橘子被夹起，或者明显被夹爪稳定夹住。
- 画面里不要有手、脸或者其他遮挡物。
- 主臂动作不要突然跳到极限位置。

如果 pilot 数据里有明显失败样本，建议直接重新录，不要拿坏数据训练。

## 3. 继续录制更多数据

pilot 数据确认没问题后，继续追加录制：

```bash
python lab/record_orange_dataset.py \
  --num-episodes 20 \
  --resume \
  --show
```

第一版建议先录：

```text
30-50 条高质量 episode
```

每条数据里可以稍微改变橘子在 wrist 画面中的位置，例如：

- 画面中心
- 略微偏左
- 略微偏右
- 稍微近一点
- 稍微远一点

但不要一开始变化太大。当前任务是“wrist-ready 后抓取橘子”，不是让模型从全桌面任意位置找橘子。

## 4. 检查数据集文件

录制完成后，检查数据集是否生成成功：

```bash
find datasets/orange_wrist_grasp_formal_clean_v1 -maxdepth 3 -type f | head -50
```

正常情况下应该能看到类似文件：

```text
meta/info.json
meta/tasks.parquet
meta/stats.json
data/chunk-000/file-000.parquet
```

如果没有 `meta/info.json`，训练脚本会拒绝启动，因为本地数据集不完整。

## 5. 开始训练

先 dry run，看训练命令是否正确：

```bash
python lab/train_orange_act.py --dry-run
```

确认没问题后开始训练：

```bash
python lab/train_orange_act.py
```

训练脚本会自动选择可用设备：

```text
cuda -> mps -> cpu
```

默认训练参数：

```text
policy: ACT
steps: 5000
batch_size: 2
save_freq: 500
log_freq: 20
dataset: datasets/orange_wrist_grasp_formal_clean_v1
```

训练开始后，终端会打印 live log 路径，例如：

```text
outputs/train/orange_act_clean_live_YYYYMMDD_HHMMSS.live.log
```

如果训练中断，可以用同一个 output dir 恢复：

```bash
python lab/train_orange_act.py \
  --resume \
  --output-dir outputs/train/orange_act_clean_live_YYYYMMDD_HHMMSS
```

## 6. 使用训练好的模型

训练完成后，模型一般在：

```text
outputs/train/<run_name>/checkpoints/last/pretrained_model
```

只测试机械臂 policy：

```bash
python lab/run_orange_act_policy.py \
  --policy-path outputs/train/<run_name>/checkpoints/last/pretrained_model \
  --reset-before \
  --final-close \
  --show \
  --execute
```

运行完整自动抓取流程：

```bash
python lab/orange_grasp_auto.py \
  --policy-path outputs/train/<run_name>/checkpoints/last/pretrained_model \
  --show \
  --execute
```

如果这个模型效果更好，并且要作为默认模型，需要更新：

```text
lab/orange_grasp_config.py
```

里面的：

```text
DEFAULT_POLICY_PATH
```

## 7. 常见问题

### 相机打不开

先检查 wrist 相机 index：

```bash
python lab/yolo_camera_check.py
```

录制脚本默认是：

```text
--wrist-camera 1
```

如果实际 wrist 相机不是 1，可以手动指定：

```bash
python lab/record_orange_dataset.py --wrist-camera <index> --show
```

### 从臂突然乱动或者跳变

立刻停止录制，检查：

- leader/follower 串口有没有接反
- 1-6 号舵机 ID 是否都能读到
- 主臂连接线是否松动
- `leader_follower_teleop.py --print-positions` 是否能稳定读取六个电机

不要把有跳变的数据录进训练集。

### 夹爪闭合不够

检查 6 号舵机范围：

```text
lab/orange_grasp_config.py
```

当前从臂夹爪范围是：

```text
open=2302
close=800
```

### 训练时去 Hugging Face 下载数据

这通常说明本地 dataset 路径错了，或者缺少 `meta/info.json`。

检查：

```bash
ls datasets/orange_wrist_grasp_formal_clean_v1/meta/info.json
```

如果这个文件不存在，说明数据集没有正确生成。

## 8. 不要做什么

录制当前橘子抓取数据集时，不要：

- 录橘子不在 wrist 画面里的 episode。
- 把底盘导航过程录进这个数据集。
- 把不同任务混在同一个数据集里。
- 在录制过程中改变相机安装位置。
- 把明显失败的抓取样本混进正常抓取数据集。

当前数据集只表达一个任务：

```text
从 wrist-ready 视角开始，用夹爪抓起橘子。
```
