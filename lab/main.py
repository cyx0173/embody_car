from __future__ import annotations

import threading
import cv2
import numpy as np

from asr import QwenASR as ASR
from nlu import NLU
from tts import TTS as TTS
from camera import CameraManager
from visual_tracking import VisualTracking
from visual_qa import VisualQA
from robotic_interaction import RoboticInteraction
from chat import ChatBot
from arm_control import ServoController


class EmbodiedAgent:
    def __init__(self):
        self.camera = CameraManager()
        self.arm = ServoController()
        self.visual_tracking = VisualTracking(camera=self.camera, arm=self.arm)
        self.robotic_interaction = RoboticInteraction(camera=self.camera, arm=self.arm)
        self.asr = ASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.visual_qa = VisualQA()
        self.chatbot = ChatBot()
        self.tracking_thread: threading.Thread | None = None

    def run(self) -> None:
        print("具身智能助手已启动，请下达指令...")
        while True:
            try:
                user_text = self.asr.listen()
                if not user_text:
                    continue

                print(f"用户: {user_text}")
                if self._is_stop_tracking_command(user_text):
                    if self._stop_tracking_if_running():
                        self._speak("好的，我已停止追踪，可以继续下达指令。")
                    else:
                        self._speak("当前没有正在进行的追踪。")
                    continue

                if self._is_exit_command(user_text):
                    self._stop_tracking_if_running()
                    self._speak("好的，已退出。")
                    break

                if self._is_reset_command(user_text):
                    self._stop_tracking_if_running()
                    self.arm.reset()
                    self._speak("好的，已重置机械臂。")
                    continue

                if not self._is_task_command(user_text):
                    self._stop_tracking_if_running()
                    reply = self._handle_chat(user_text)
                    self._speak(reply)
                    continue

                parsed = self.nlu.predict(user_text)
                print(f"NLU: {parsed}")

                if not parsed.get("valid", True):
                    self._speak(self._format_nlu_error(parsed))
                    continue

                intent = parsed.get("intent")
                target = parsed.get("target")
                self._stop_tracking_if_running()
                self.prepare_for_intent(intent)

                if intent == "visual_tracking":
                    reply = self._handle_tracking(target, None)
                    self._speak(reply)
                elif intent == "visual_understanding":
                    current_image = self._capture_image()
                    reply = self._handle_vqa(user_text, current_image)
                    if reply:
                        self._speak(reply)
                elif intent == "object_interaction":
                    reply = self._handle_interaction(target)
                    self._speak(reply)
                elif intent == "voice_chat":
                    reply = self._handle_chat(user_text)
                    self._speak(reply)
                elif intent == "reset_arm":
                    self.arm.reset()
                    self._speak("好的，已重置机械臂。")
                else:
                    self._speak("未能识别有效意图，请重新下达指令。")
            except KeyboardInterrupt:
                print("\n已退出。")
                break
            except Exception as exc:
                print(f"执行异常: {exc}")
                self._speak("执行过程中遇到问题，请检查目标、相机或机械臂状态。")

    def _capture_image(self) -> np.ndarray:
        frame = self.camera.read_hand_raw()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _format_nlu_error(self, parsed: dict) -> str:
        if parsed.get("error") == "missing_target":
            return "我还不知道要操作哪个目标，请说清楚目标物体。"
        return "指令解析不完整，请重新说一遍。"

    def _is_exit_command(self, text: str) -> bool:
        return any(word in text for word in ("退出", "结束程序", "关闭程序", "拜拜", "再见"))

    def _is_stop_tracking_command(self, text: str) -> bool:
        stop_words = ("停止", "停下", "结束", "别追", "不要追", "停止追踪", "停止跟随")
        tracking_words = ("追踪", "跟随", "跟着", "看着", "盯着", "tracking")
        return any(word in text for word in stop_words) and (
            self._tracking_is_running() or any(word in text for word in tracking_words)
        )

    def _is_reset_command(self, text: str) -> bool:
        return any(word in text for word in ("复位", "重置", "reset", "回到初始", "回到原位"))

    def _is_task_command(self, text: str) -> bool:
        tracking_words = ("追踪", "跟随", "跟着", "看着", "盯着", "看我的")
        interaction_words = ("抓", "拿", "碰", "触碰", "点", "按", "推", "移动到", "靠近")
        visual_words = (
            "画面",
            "图片",
            "图里",
            "镜头",
            "相机",
            "看到",
            "看见",
            "前面",
            "这里",
            "那边",
            "我正在",
            "电脑上",
            "屏幕",
            "做什么",
        )
        target_words = (
            "人",
            "瓶子",
            "杯子",
            "电脑",
            "屏幕",
            "手机",
            "键盘",
            "鼠标",
            "椅子",
            "书",
        )

        if any(word in text for word in tracking_words + interaction_words + visual_words):
            return True
        return any(action in text for action in ("找", "寻找")) and any(
            target in text for target in target_words
        )

    def _speak(self, text: str) -> None:
        print(f"助手: {text}")
        self.tts.speak(text)

    def _tracking_is_running(self) -> bool:
        return self.tracking_thread is not None and self.tracking_thread.is_alive()

    def _stop_tracking_if_running(self) -> bool:
        if self.visual_tracking is not None and self._tracking_is_running():
            print("检测到正在追踪，先停止追踪")
            self.visual_tracking.stop()
            self.tracking_thread.join(timeout=5.0)
            if self.tracking_thread.is_alive():
                print("追踪线程仍未退出，将继续在后台尝试停止")
                return False
            else:
                self.tracking_thread = None
                return True
        self.tracking_thread = None
        return False

    def prepare_for_intent(self, intent: str | None) -> None:
        if intent == "object_interaction":
            self.arm.reset()
        elif intent == "visual_tracking":
            self.arm.reset()
        elif intent == "reset_arm":
            self.arm.reset()

    def _handle_tracking(self, target: str | None, _current_image) -> str | None:
        if not target:
            return "请告诉我要追踪哪个目标。"
        self._stop_tracking_if_running()
        self.tracking_thread = threading.Thread(
            target=self.visual_tracking.track,
            args=(target,),
            daemon=True,
        )
        self.tracking_thread.start()
        return f"我开始追踪{target}了。"

    def _handle_vqa(self, user_text: str, current_image: np.ndarray) -> str | None:
        return self.visual_qa.answer(current_image, user_text)

    def _handle_chat(self, user_text: str) -> str | None:
        return self.chatbot.reply(user_text)

    def _handle_interaction(self, target: str | None) -> str | None:
        if not target:
            return "请告诉我要操作哪个目标。"
        self._speak(f"好的，我开始寻找{target}，请注意机械臂移动。")
        try:
            self.robotic_interaction.interact(target)
        finally:
            self.robotic_interaction.state = "IDLE"
        return f"我已经移动到{target}附近了，可以继续和我说话。"


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
