from __future__ import annotations

from arm_control import ServoController


class RoboticInteraction:
    """机器人交互模块"""

    def __init__(self, dry_run: bool = False, port: str | None = None):
        self.dry_run = dry_run
        self.port = port
        self._arm: ServoController | None = None

    @property
    def arm(self) -> ServoController:
        if self._arm is None:
            if self.port:
                self._arm = ServoController(port=self.port)
            else:
                self._arm = ServoController()
        return self._arm

    def interact(self, target, image, detections=None):
        """根据目标与图像信息执行机器人交互"""
        if self.dry_run:
            if target:
                return f"已理解动作目标是 {target}，当前处于dry-run模式，不实际驱动机械臂。"
            return "已理解机械臂交互指令，当前处于dry-run模式，不实际驱动机械臂。"

        if target in ("复位", "home", "reset") or target is None:
            self.arm.reset()
            return "机械臂已复位。"

        if detections:
            match = next((det for det in detections if det.label == target), None)
            if match:
                cx, cy = match.center
                return f"已看到 {target}，目标中心在图像坐标 ({cx:.0f}, {cy:.0f})。抓取还需要手眼标定和末端执行器控制。"

        return f"还没有在画面中稳定定位到 {target}，暂不执行机械臂动作。"

    def close(self):
        if self._arm is not None:
            self._arm.close()
            self._arm = None
