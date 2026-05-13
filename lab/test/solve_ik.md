ik_solver.py          # 唯一的文件
  ├── Joint           # 数据类：关节参数（xyz, rpy, axis, limits）
  ├── FK相关的数学函数
  │     ├── rpy_to_matrix()
  │     ├── make_transform()
  │     └── compute_fk() / get_tip_position()
  ├── Jacobian
  │     └── geometric_jacobian()
  ├── IK核心
  │     ├── clip_to_limits()
  │     └── solve_position_ik()    # 阻尼最小二乘迭代
  └── 你自己的机械臂参数
        └── 直接硬编码进文件里（不依赖URDF）

def solve_ik(self, target_world_xyz: np.ndarray, initial_joint_values: dict[str, float] | None = None) -> tuple[dict[str, float], float]:
A. URDF 解析 — parse_urdf_like()，从 xacro 读关节参数
B. FK（正运动学） — compute_fk()，给定关节角算末端位置
C. 几何 Jacobian — geometric_jacobian()，求末端速度对关节角的偏导
D. IK 求解器 — solve_position_ik()，迭代法求逆解