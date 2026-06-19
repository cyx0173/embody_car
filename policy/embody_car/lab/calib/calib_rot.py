import cv2
import numpy as np

# 1. 准备棋盘格参数
board_size = (9, 6)  # 棋盘格内角点数量
square_size = 0.01688  # 每个格子的物理尺寸 (米)

# 2. 准备容器存储角点数据
obj_pts = [] # 3D 空间点
img_pts_l = [] # 左图 2D 像素点
img_pts_r = [] # 右图 2D 像素点

# 准备 3D 参考坐标 (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

# 3. 假设你已经有了左右相机的内参 (从你的 JSON 中读取)
# K1, dist1 = ... (左相机)
# K2, dist2 = ... (右相机)

# 4. 读取图片并寻找角点
# 这里循环读取你拍好的 20 对双目照片
for i in range(20):
    img_l = cv2.imread(f'left_{i}.jpg')
    img_r = cv2.imread(f'right_{i}.jpg')
    
    ret_l, corners_l = cv2.findChessboardCorners(img_l, board_size)
    ret_r, corners_r = cv2.findChessboardCorners(img_r, board_size)
    
    if ret_l and ret_r:
        obj_pts.append(objp)
        img_pts_l.append(corners_l)
        img_pts_r.append(corners_r)

# 5. 【核心步骤】双目联合标定
# 我们固定内参 (FIX_INTRINSIC)，只求相对位置 R 和 t
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    obj_pts, img_pts_l, img_pts_r, 
    K1, dist1, K2, dist2, 
    img_l.shape[:2][::-1], 
    criteria=criteria, flags=flags
)

print("--- 相对位置获取成功 ---")
print("旋转矩阵 R:\n", R)
print("平移向量 T (单位:米):\n", T)
print("重投影误差 (越小越好):", ret)