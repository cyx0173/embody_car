from __future__ import annotations

import argparse
import time
from pathlib import Path

from nlu import NLU


class ConsoleTTS:
    def speak(self, text: str):
        print(f"助手: {text}", flush=True)


class EmbodiedAgent:
    def __init__(
        self,
        camera_id: int = 0,
        yolo_model: str | Path | None = None,
        no_tts: bool = False,
        dry_run: bool = True,
        arm_port: str | None = None,
        asr_device: int | None = None,
        asr_timeout: float = 12.0,
        max_end_silence: int = 1000,
        show_camera: bool = False,
    ):
        base_dir = Path(__file__).resolve().parent
        self.camera_id = camera_id
        self.yolo_model = yolo_model or base_dir / "yolo11n.pt"
        self.nlu = NLU()
        self.no_tts = no_tts
        self.dry_run = dry_run
        self.arm_port = arm_port
        self.asr_device = asr_device
        self.asr_timeout = asr_timeout
        self.max_end_silence = max_end_silence
        self.show_camera = show_camera
        self._camera = None
        self._perception = None
        self._visual_qa = None
        self._asr = None
        self._tts = ConsoleTTS() if no_tts else None
        self._visual_tracking = None
        self._robotic_interaction = None

    @property
    def camera(self):
        if self._camera is None:
            print("[INIT] 正在打开摄像头...", flush=True)
            from perception import Camera

            self._camera = Camera(camera_id=self.camera_id)
        return self._camera

    @property
    def perception(self):
        if self._perception is None:
            print("[INIT] 正在加载YOLO模型...", flush=True)
            from perception import YoloPerception

            self._perception = YoloPerception(self.yolo_model)
        return self._perception

    @property
    def visual_qa(self):
        if self._visual_qa is None:
            print("[INIT] 正在初始化视觉语言模型...", flush=True)
            from visual_qa import VisualQA

            self._visual_qa = VisualQA()
        return self._visual_qa

    @property
    def asr(self):
        if self._asr is None:
            print("[INIT] 正在初始化语音识别...", flush=True)
            from asr import QwenASR as ASR

            self._asr = ASR(
                input_device_index=self.asr_device,
                timeout_seconds=self.asr_timeout,
                max_end_silence=self.max_end_silence,
            )
        return self._asr

    @property
    def tts(self):
        if self._tts is None:
            print("[INIT] 正在初始化语音播报...", flush=True)
            from tts import TTS

            self._tts = TTS()
        return self._tts

    @property
    def visual_tracking(self):
        if self._visual_tracking is None:
            from visual_tracking import VisualTracking

            self._visual_tracking = VisualTracking()
        return self._visual_tracking

    @property
    def robotic_interaction(self):
        if self._robotic_interaction is None:
            from robotic_interaction import RoboticInteraction

            self._robotic_interaction = RoboticInteraction(dry_run=self.dry_run, port=self.arm_port)
        return self._robotic_interaction

    def run(self):
        print("具身智能助手已启动，请下达语音指令。Ctrl+C退出。", flush=True)
        while True:
            try:
                user_text = self.asr.listen({"is_speaking": False})
                if user_text:
                    self.handle_text(user_text)
                else:
                    print("[ASR] 没有识别到有效语音，继续监听。", flush=True)
            except KeyboardInterrupt:
                print("\n已退出。")
                break
        self.close()

    def run_voice_once(self):
        print("单轮语音模式：请说一句指令，例如“我正在电脑上做什么”。", flush=True)
        try:
            user_text = self.asr.listen({"is_speaking": False})
            if user_text:
                self.handle_text(user_text)
            else:
                print("[ASR] 本轮没有识别到有效语音。", flush=True)
        finally:
            self.close()

    def run_console(self):
        print("文本调试模式。输入 q/quit/exit 退出。", flush=True)
        while True:
            user_text = input("用户: ").strip()
            if user_text.lower() in {"q", "quit", "exit"}:
                break
            if user_text:
                self.handle_text(user_text)
        self.close()

    def handle_text(self, user_text: str) -> str:
        print(f"用户: {user_text}", flush=True)
        parsed_data = self.nlu.parse(user_text)
        intent = parsed_data.get("intent")
        target = parsed_data.get("target_object")

        print("[VISION] 正在采集当前画面...", flush=True)
        current_image = self._capture_image()
        print("[VISION] 正在运行YOLO检测...", flush=True)
        detections = self.perception.detect(current_image, target=None)
        from perception import draw_detections, show_image, summarize_detections

        detection_context = summarize_detections(detections)
        print(detection_context, flush=True)
        if self.show_camera:
            annotated = draw_detections(current_image, detections)
            show_image("Robot camera - YOLO detections", annotated, wait_ms=1)

        if intent == "visual_tracking":
            reply = self._handle_tracking(target)
        elif intent == "visual_qa":
            reply = self._handle_vqa(user_text, current_image, detection_context)
        elif intent == "robotic_interaction":
            reply = self._handle_interaction(target, current_image, detections)
        else:
            reply = "未能识别有效意图，请重新下达指令。"

        self.tts.speak(reply)
        return reply

    def _capture_image(self):
        return self.camera.capture()

    def _handle_tracking(self, target):
        if not target:
            return "请告诉我要追踪哪个目标，例如人、杯子或手机。"
        if self.dry_run:
            return f"已理解需要追踪 {target}。当前是dry-run模式，不驱动机械臂。"
        self.visual_tracking.track(target)
        return "目标追踪完成。"

    def _handle_vqa(self, user_text, current_image, detection_context):
        return self.visual_qa.answer(current_image, user_text, context=detection_context)

    def _handle_interaction(self, target, current_image, detections):
        return self.robotic_interaction.interact(target, current_image, detections=detections)

    def close(self):
        if self._camera is not None:
            self._camera.close()
        if self._robotic_interaction is not None:
            self._robotic_interaction.close()
        if self.show_camera:
            from perception import close_windows

            close_windows()


def preview_camera(camera_id: int, yolo_model: str | None = None):
    from perception import Camera, YoloPerception, draw_detections, show_image

    base_dir = Path(__file__).resolve().parent
    camera = Camera(camera_id=camera_id)
    perception = YoloPerception(yolo_model or base_dir / "yolo11n.pt")
    print("摄像头预览已启动，按 q 退出。", flush=True)
    try:
        while True:
            frame = camera.capture()
            detections = perception.detect(frame)
            annotated = draw_detections(frame, detections)
            show_image("Robot camera preview", annotated, wait_ms=1)
            import cv2

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        camera.close()
        from perception import close_windows

        close_windows()


def build_parser():
    parser = argparse.ArgumentParser(description="Embodied vision-language-action agent")
    parser.add_argument("--text", action="store_true", help="使用键盘输入代替麦克风，便于先调通链路")
    parser.add_argument("--once", type=str, default=None, help="执行一条文本指令后退出")
    parser.add_argument("--voice-once", action="store_true", help="听一条语音指令、处理并退出，便于调试")
    parser.add_argument("--preview-camera", action="store_true", help="打开摄像头预览窗口并显示YOLO检测框")
    parser.add_argument("--show-camera", action="store_true", help="处理每条指令时显示当前图像和YOLO框")
    parser.add_argument("--list-audio-devices", action="store_true", help="列出可用麦克风输入设备后退出")
    parser.add_argument("--asr-device", type=int, default=None, help="指定PyAudio输入设备index")
    parser.add_argument("--asr-timeout", type=float, default=12.0, help="单轮语音监听最长秒数")
    parser.add_argument("--max-end-silence", type=int, default=1000, help="句尾静音判定，单位毫秒")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV摄像头编号")
    parser.add_argument("--yolo-model", type=str, default=None, help="YOLO模型路径，默认使用lab/yolo11n.pt")
    parser.add_argument("--no-tts", action="store_true", help="只在终端打印回复，不加载TTS模型")
    parser.add_argument("--execute", action="store_true", help="实际驱动追踪/机械臂；默认dry-run")
    parser.add_argument("--arm-port", type=str, default=None, help="机械臂串口，例如/dev/cu.usbmodem...")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.list_audio_devices:
        from asr import list_input_devices

        devices = list_input_devices()
        if not devices:
            print("未发现可用输入设备。")
        for dev in devices:
            print(
                f"{dev['index']}: {dev['name']} "
                f"(channels={dev['channels']}, default_sr={dev['sample_rate']})"
            )
        raise SystemExit(0)
    if args.preview_camera:
        preview_camera(args.camera_id, args.yolo_model)
        raise SystemExit(0)

    agent = EmbodiedAgent(
        camera_id=args.camera_id,
        yolo_model=args.yolo_model,
        no_tts=args.no_tts,
        dry_run=not args.execute,
        arm_port=args.arm_port,
        asr_device=args.asr_device,
        asr_timeout=args.asr_timeout,
        max_end_silence=args.max_end_silence,
        show_camera=args.show_camera,
    )
    if args.once:
        agent.handle_text(args.once)
        agent.close()
    elif args.voice_once:
        agent.run_voice_once()
    elif args.text:
        agent.run_console()
    else:
        agent.run()
