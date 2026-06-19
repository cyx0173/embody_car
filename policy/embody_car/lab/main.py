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
from clean_pick_place_skill import CleanPickPlaceSkill


class EmbodiedAgent:
    def __init__(self):
        self.camera = CameraManager()
        self.visual_tracking = VisualTracking(camera=self.camera)
        self.robotic_interaction = RoboticInteraction(camera=self.camera)
        self.asr = ASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.visual_qa = VisualQA()
        self.chatbot = ChatBot()
        self.pick_place = CleanPickPlaceSkill()
        self.tracking_thread: threading.Thread | None = None

    def run(self) -> None:
        print("具身智能助手已启动，请下达指令...")
        while True:
            try:
                user_text = self.asr.listen()
                if not user_text:
                    continue

                print(f"用户: {user_text}")
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
                elif intent == "object_grasp":
                    reply = self._handle_object_grasp(parsed)
                    self._speak(reply)
                elif intent == "voice_chat":
                    reply = self._handle_chat(user_text)
                    self._speak(reply)
                elif intent == "reset_arm":
                    self.robotic_interaction.arm.reset()
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
            pass
        elif intent == "object_grasp":
            pass
        elif intent == "visual_tracking":
            pass
        elif intent == "reset_arm":
            self.robotic_interaction.arm.reset()

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

    def _close_serial_owner(self, owner) -> None:
        arm = getattr(owner, "arm", None)
        serial_obj = getattr(arm, "_ser", None)
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass

    def _release_resources_for_pick_place(self) -> None:
        self._stop_tracking_if_running()
        self._close_serial_owner(self.visual_tracking)
        self._close_serial_owner(self.robotic_interaction)
        self.camera.release()

    def _restore_shared_modules(self) -> None:
        self.camera = CameraManager()
        self.visual_tracking = VisualTracking(camera=self.camera)
        self.robotic_interaction = RoboticInteraction(camera=self.camera)
        self.tracking_thread = None

    def _target_bowl_from_nlu(self, parsed: dict) -> str:
        destination = parsed.get("destination")
        destination_qualifier = parsed.get("destination_qualifier")
        target_qualifier = parsed.get("target_qualifier")

        if destination == "bowl" and destination_qualifier in {"left", "right"}:
            return destination_qualifier
        if target_qualifier in {"left", "right"}:
            return target_qualifier
        return "right"

    def _handle_object_grasp(self, parsed: dict) -> str | None:
        target = parsed.get("target")
        if not target:
            return "请告诉我要抓取哪个目标。"

        target_bowl = self._target_bowl_from_nlu(parsed)
        self._speak(f"好的，我开始抓取{target}，准备放到{target_bowl}边的碗里。")
        self._release_resources_for_pick_place()
        try:
            return self.pick_place.run(target=target, target_bowl=target_bowl)
        finally:
            self._restore_shared_modules()


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
