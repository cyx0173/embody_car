import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

JAW_LINK_LENGTH_M = 0.083
DEFAULT_JAW_TIP_LOCAL_XYZ = np.array([JAW_LINK_LENGTH_M, 0.0, 0.0], dtype=float)

@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


DEFAULT_ZERO_POSE = {
    "gripper": 0.0,
    "wrist_roll": 0.0,
    "wrist_flex": 0,
    "elbow_flex": 0,
    "shoulder_lift": 0,
    "shoulder_pan": 0.0,
}

IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SIM_JOINT_OFFSETS = {
    "gripper":- math.pi / 2,
}
SIM_JOINT_DIRECTIONS = {}

def point_in_link_frame(link_frame: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = local_point
    return (link_frame @ homogeneous)[:3]

def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4, dtype=float)

    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1 - c

    rot = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot
    return transform

def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx

def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform

def compute_fk(
    root_link: str,
    children_map: dict[str, list[Joint]],
    joint_values: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    link_frames: dict[str, np.ndarray] = {root_link: np.eye(4, dtype=float)}
    joint_frames: dict[str, np.ndarray] = {}

    def visit(link_name: str) -> None:
        parent_frame = link_frames[link_name]
        for joint in children_map.get(link_name, []):
            origin_tf = make_transform(joint.xyz, joint.rpy)
            motion_tf = np.eye(4, dtype=float)

            if joint.joint_type in {"revolute", "continuous"}:
                sim_angle = (
                    joint_values.get(joint.name, 0.0) * SIM_JOINT_DIRECTIONS.get(joint.name, 1.0)
                    + SIM_JOINT_OFFSETS.get(joint.name, 0.0)
                )
                motion_tf = rotation_about_axis(joint.axis, sim_angle)

            joint_frame = parent_frame @ origin_tf
            child_frame = joint_frame @ motion_tf
            joint_frames[joint.name] = joint_frame
            link_frames[joint.child] = child_frame
            visit(joint.child)

    visit(root_link)
    return link_frames, joint_frames

def clip_to_limits(joint_values: dict[str, float], joint_by_name: dict[str, Joint]) -> dict[str, float]:
    clipped = dict(joint_values)
    for name, value in clipped.items():
        joint = joint_by_name[name]
        if joint.lower is not None:
            value = max(value, joint.lower)
        if joint.upper is not None:
            value = min(value, joint.upper)
        clipped[name] = value
    return clipped
  
def get_tip_position(
    root_link: str,
    children_map: dict[str, list[Joint]],
    joint_values: dict[str, float],
    tip_link: str,
    tip_offset: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    link_frames, joint_frames = compute_fk(root_link, children_map, joint_values)
    tip_position = point_in_link_frame(link_frames[tip_link], tip_offset)
    return tip_position, link_frames, joint_frames

def geometric_jacobian(
    root_link: str,
    children_map: dict[str, list[Joint]],
    joint_by_name: dict[str, Joint],
    joint_names: list[str],
    joint_values: dict[str, float],
    tip_link: str,
    tip_offset: np.ndarray,
) -> np.ndarray:
    tip_position, _, joint_frames = get_tip_position(root_link, children_map, joint_values, tip_link, tip_offset)
    jacobian = np.zeros((3, len(joint_names)), dtype=float)

    for index, joint_name in enumerate(joint_names):
        joint = joint_by_name[joint_name]
        joint_origin = joint_frames[joint_name][:3, 3]
        axis_local = joint.axis
        axis_norm = np.linalg.norm(axis_local)
        if axis_norm == 0:
            continue
        axis_world = joint_frames[joint_name][:3, :3] @ (axis_local / axis_norm)
        jacobian[:, index] = np.cross(axis_world, tip_position - joint_origin)

    return jacobian

def solve_position_ik(
    root_link: str,
    children_map: dict[str, list[Joint]],
    joint_by_name: dict[str, Joint],
    target: np.ndarray,
    tip_link: str,
    tip_offset: np.ndarray,
    solve_joint_names: list[str],
    initial_joint_values: dict[str, float] | None = None,
    max_iterations: int = 1000000000,
    tolerance: float = 1e-4,
    damping: float = 1e-4,
    max_step: float = 0.35,
) -> tuple[dict[str, float], float]:
    joint_values = {
        name: (
            initial_joint_values[name]
            if initial_joint_values is not None and name in initial_joint_values
            else DEFAULT_ZERO_POSE.get(name, 0.0)
        )
        for name in solve_joint_names
    }
    joint_values = clip_to_limits(joint_values, joint_by_name)
    best_values = dict(joint_values)
    best_error = float("inf")
    count = 0

    for _ in range(max_iterations):
        count += 1
        current_position, link_frames, joint_frames = get_tip_position(
            root_link, children_map, joint_values, tip_link, tip_offset
        )
        error = target - current_position
        error_norm = float(np.linalg.norm(error))

        if error_norm < best_error:
            best_error = error_norm
            best_values = dict(joint_values)

        if error_norm < tolerance:
            return joint_values, error_norm
        #print(error_norm)
        jacobian = geometric_jacobian(
            root_link, children_map, joint_by_name, solve_joint_names, joint_values, tip_link, tip_offset
        )
        normal_matrix = jacobian.T @ jacobian + damping * np.eye(len(solve_joint_names), dtype=float)
        step = np.linalg.solve(normal_matrix, jacobian.T @ error)

        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step = step * (max_step / step_norm)

        candidate = dict(joint_values)
        for name, delta in zip(solve_joint_names, step):
            candidate[name] += float(delta)
        joint_values = clip_to_limits(candidate, joint_by_name)
    print(f"iterations: {count}")
    return best_values, best_error

def parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)
    
def parse_urdf_like(path: str) -> tuple[list[str], list[Joint]]:
    tree = ET.parse(path)
    root = tree.getroot()

    links = [link.attrib["name"] for link in root.findall("link")]
    joints: list[Joint] = []

    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        parent = joint.find("parent")
        child = joint.find("child")

        joints.append(
            Joint(
                name=joint.attrib["name"],
                joint_type=joint.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
                rpy=parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
                axis=parse_xyz(axis.attrib.get("xyz") if axis is not None else "1 0 0"),
                lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
                upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            )
        )

    return links, joints

def build_tree(joints: list[Joint]) -> tuple[dict[str, list[Joint]], set[str], set[str], dict[str, Joint]]:
    children: dict[str, list[Joint]] = {}
    parent_links = set()
    child_links = set()
    joint_by_name: dict[str, Joint] = {}

    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)
        parent_links.add(joint.parent)
        child_links.add(joint.child)
        joint_by_name[joint.name] = joint

    return children, parent_links, child_links, joint_by_name

class RobotIKSolver:
    def __init__(
        self,
        xacro_path: str | Path | None = None,
    ) -> None:
        self.xacro_path = xacro_path or "lab/interaction/so101_follower.urdf.xacro"
        _, joints = parse_urdf_like(str(self.xacro_path))
        self.joints = joints
        self.children_map, parent_links, child_links, self.joint_by_name = build_tree(joints)
        roots = sorted(parent_links - child_links)
        self.root_link = "world" if "world" in parent_links else roots[0]
        self.current_joint_values: dict[str, float] = {
            name: DEFAULT_ZERO_POSE.get(name, 0.0) for name in IK_JOINT_NAMES
        }

    def solve_ik(self, target_world_xyz: np.ndarray, initial_joint_values: dict[str, float] | None = None) -> tuple[dict[str, float], float]:
        initial = dict(self.current_joint_values if initial_joint_values is None else initial_joint_values)
        solved_subset, error = solve_position_ik(
            root_link=self.root_link,
            children_map=self.children_map,
            joint_by_name=self.joint_by_name,
            target=np.asarray(target_world_xyz, dtype=float),
            tip_link="jaw_link",
            tip_offset=DEFAULT_JAW_TIP_LOCAL_XYZ,
            solve_joint_names=IK_JOINT_NAMES,
            initial_joint_values=initial,
        )
        solved = dict(initial)
        solved.update({k: float(v) for k, v in solved_subset.items()})
        self.current_joint_values = solved
        return solved, float(error)

"""
def solve_ik(self) -> None:
        直接复用 RobotController.solve_ik()
        assert self.selected_row is not None, "call auto_select() first"
        target = np.array(
            [self.selected_row["x"], self.selected_row["y"], self.selected_row["z"]],
            dtype=float,
        )
        self.last_ik_solution, self.last_ik_error = self.robot.solve_ik(target)
        print(f"[Step 5] IK solved. error={self.last_ik_error:.4f} m, joints={self.last_ik_solution}")
"""