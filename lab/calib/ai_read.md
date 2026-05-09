# 🤖 具身智能双目视觉系统开发指南 (Stereo Vision Pipeline Spec)

## 1. 项目背景与硬件
- **目标**：机械臂系统开发无尺寸预设的“双目测距”与“雅可比矩阵运动控制”流水线。
- **相机配置**：
  - 相机 0 (Left/Base): 机械臂底座主视角 `cap_id=0`
  - 相机 1 (Right/Aux): 侧边辅助视角 `cap_id=1`
- **画面规格**：
  - 物理原始分辨率：1920 x 1080
  - **核心约束（Working Resolution）**：必须经过 `center_crop` (1080x1080) 然后 `resize` 到 540x540。所有特征点提取和标定都必须在 **540x540** 上进行。

## 2. 核心工作流 (The 3-Step Pipeline)

### Step 1: 内参标定 (Intrinsic Calibration) - ✅ [已完成]
- **脚本**: `calibrate_intrinsics.py`
- **产出**: `camera_left.json` (对应 cam 0) 和 `camera_right.json` (对应 cam 1)。
- **注意**: 此步骤计算出的相机内参 (camera_matrix) 是基于 540x540 的画面。

### Step 2: 双目同步采集 (Stereo Image Capture) - ⏳ [待开发]
- **目标脚本**: `capture_stereo.py`
- **功能**: 同时打开 `cap(0)` 和 `cap(1)`，实时应用 540x540 的裁剪逻辑，展示拼接画面。当检测到棋盘格且按下 `s` 键时，同步保存左右帧用于后续外参计算。

### Step 3: 双目相对位置标定 (Stereo Extrinsic Calibration) - ⏳ [待开发]
- **目标脚本**: `calibrate_stereo.py`
- **功能**: 读取 Step 1 的两个 JSON 内参文件，读取 Step 2 拍下的照片对。
- **核心算法**: 运行 `cv2.stereoCalibrate` (开启 `CALIB_FIX_INTRINSIC` 标志)。
- **产出**: 计算并保存两个相机的相对位置：旋转矩阵 $R$、平移向量 $t$（含基线距离），以及用于 3D 还原的投影矩阵 $P_1, P_2$，存为 `stereo_extrinsics.json`。

## 3. 代码编写规范 (Coding Guidelines)
1. **严格继承裁剪逻辑**: AI 在编写 Step 2 和 Step 3 时，必须复用 `calibrate_intrinsics.py` 中的 `center_crop_and_resize_frame` 函数。
2. **格式统一**: 使用 Python 3.10+ 类型提示 (Type hints)，使用 `argparse` 处理命令行参数，使用 `pathlib` 处理路径。
3. **坐标单位**: 物理距离严格使用**米 (meters)**。

## 4. 当前任务 (Current Task)
AI 助手，请阅读上述上下文，并在接下来的对话中，根据我的指令依次实现 `capture_stereo.py` 和 `calibrate_stereo.py`。

## 5. 标定参数

棋盘格方块边长：**2.54 cm = 0.0254 m**（已实测）

## 6. 标定命令

### Step 1: 内参标定（左相机）

```bash
cd /Users/chengyx/Code/具身智能引论/lab/calib
python calibrate_intrinsics.py --camera 0 --cols 9 --rows 6 --square-size 0.0254 --output camera_left.json
```

### Step 2: 内参标定（右相机）

```bash
python calibrate_intrinsics.py --camera 1 --cols 9 --rows 6 --square-size 0.0254 --output camera_right.json
```

### Step 3: 双目同步采集

```bash
python capture_stereo.py --cap0 0 --cap1 1 --cols 9 --rows 6 --square-size 0.0254
```

### Step 4: 双目外参标定

```bash
python calibrate_stereo.py \
    --left-intrinsics camera_left.json \
    --right-intrinsics camera_right.json \
    --stereo-pairs ./stereo_captures \
    --cols 9 --rows 6 --square-size 0.0254
```

## 5. 核心代码伪代码细节 (Pseudo-Logic Details)

### Step 2: capture_stereo.py 逻辑规范
AI 编写此脚本时必须遵循以下执行流：
1. **初始化**: 同时初始化 `cap0 = cv2.VideoCapture(0)` 和 `cap1 = cv2.VideoCapture(1)`。
2. **循环读取**:
    - 同步抓取帧：`ret0, frame0 = cap0.read()`, `ret1, frame1 = cap1.read()`。
    - **必须调用** `center_crop_and_resize_frame` 处理两路画面至 540x540。
    - 将两路画面转为灰度图 `gray0`, `gray1`。
3. **双目实时检测**:
    - 同时在 `gray0` 和 `gray1` 调用 `findChessboardCorners`。
    - 只有当 **两路画面同时检测到棋盘格** 时，才进行 `cornerSubPix` 亚像素精细化。
    - 在实时显示窗口（通过 `np.hstack` 拼接展示）中用 `drawChessboardCorners` 反馈检测状态。
4. **保存逻辑**:
    - 当用户按下 `s` 键且 `both_detected == True`：
    - 将 **原始裁剪后的 540x540 帧**（不含绘图标记）分别保存为 `data/left_idx.jpg` 和 `data/right_idx.jpg`。

### Step 3: calibrate_stereo.py 逻辑规范
AI 编写此脚本时必须遵循以下解算流：
1. **加载内参**:
    - 从 `camera_left.json` 加载 `K_L` (camera_matrix) 和 `D_L` (dist_coeffs)。
    - 从 `camera_right.json` 加载 `K_R` 和 `D_R`。
2. **准备匹配数据**:
    - 遍历 `data/` 文件夹，读取成对的 `left_XX.jpg` 和 `right_XX.jpg`。
    - 提取每对图像的棋盘格角点。
3. **双目解算 (Stereo Calibration)**:
    - 调用 `cv2.stereoCalibrate`。
    - **关键 Flag**: 必须设置 `cv2.CALIB_FIX_INTRINSIC`。这意味着我们完全信任 Step 1 的内参，只计算两台相机之间的 $R$ 和 $t$。
4. **结果持久化**:
    - 保存 `stereo_extrinsics.json`，必须包含以下字段：
        - `R`: 3x3 旋转矩阵 (list)
        - `T`: 3x1 平移向量 (list)
        - `E`: 本质矩阵
        - `F`: 基础矩阵
        - `P1, P2`: 用于三角化的 3x4 投影矩阵 (Projection Matrices)