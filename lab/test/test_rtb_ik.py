import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
import numpy as np

# 最终对齐版 URDF：修正了底座朝向，放宽了手腕限位
URDF_CONTENT = """
<robot name="so101">
  <link name="base_link"/>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0.02079 -0.02307 0.09488" rpy="0 0 3.14159"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="shoulder_link"/>
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="-0.03039 -0.01827 -0.0542" rpy="-1.5708 -1.5708 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="10"/>
  </joint>
  <link name="upper_arm_link"/>
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="lower_arm_link"/>
    <origin xyz="-0.11257 -0.028 0" rpy="0 0 1.5708"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="10"/>
  </joint>
  <link name="lower_arm_link"/>
  <joint name="wrist_flex" type="revolute">
    <parent link="lower_arm_link"/>
    <child link="wrist_link"/>
    <origin xyz="-0.1349 0.0052 0" rpy="0 0 -1.5708"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="wrist_link"/>
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist_link"/>
    <child link="jaw_link"/>
    <origin xyz="0 -0.0611 0.0181" rpy="1.5708 0 3.14159"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="jaw_link"/>
</robot>
"""

def solve_ik(target_xyz):
    model = pin.buildModelFromXML(URDF_CONTENT)
    data = model.createData()
    configuration = pink.Configuration(model, data, pin.neutral(model))

    tasks = {
        # orientation_cost 设为 0，让手腕根据位置自动寻找最自然姿态
        "tip": FrameTask("jaw_link", position_cost=1.0, orientation_cost=0.0),
        "posture": PostureTask(cost=1e-3)
    }

    tasks["tip"].set_target(pin.SE3(np.eye(3), np.array(target_xyz)))
    tasks["posture"].set_target(pin.neutral(model))

    dt = 0.05
    for _ in range(50):
        velocity = pink.solve_ik(configuration, tasks.values(), dt, solver="quadprog")
        configuration.integrate_inplace(velocity, dt)
        err = np.linalg.norm(tasks["tip"].compute_error(configuration)[:3])
        if err < 1e-4:
            break

    return {model.names[i]: float(configuration.q[i-1]) for i in range(1, model.nq + 1)}, err

if __name__ == "__main__":
    # 目标：前方 20cm，横向 0，高度 10cm
    res, final_err = solve_ik([1, 0.5, 0.1])
    print(f"解算结果 (误差: {final_err:.6f}m):")
    for name, angle in res.items():
        print(f"  {name}: {angle:.4f} rad")