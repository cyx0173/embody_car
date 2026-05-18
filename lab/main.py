from asr import QwenASR as ASR
from nlu import NLU
from tts import TTS as TTS

import cv2
import numpy as np
import threading


class EmbodiedAgent:
    def __init__(self):
        self.asr = ASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.visual_tracking = None
        self.visual_qa = None
        self.robotic_interaction = None
        self.chatbot = None
        self.camera = None
        self.camera_id = 0
        self.tracking_thread = None

    def run(self):
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

                if not self._is_task_command(user_text):
                    self._stop_tracking_if_running()
                    self._reset_active_arm_if_needed(required=False)
                    reply = self._handle_chat(user_text)
                    self._speak(reply)
                    continue

                parsed_data = self.nlu.predict(user_text)
                print(f"NLU: {parsed_data}")
                if not parsed_data.get("valid", True):
                    self._speak(self._format_nlu_error(parsed_data))
                    continue

                intent = parsed_data.get("intent")
                target = parsed_data.get("target")
                if target == 'apple':
                    target = 'orange'
                self._stop_tracking_if_running()
                self._reset_arm_before_intent(intent)

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
                else:
                    self._speak("未能识别有效意图，请重新下达指令。")
            except KeyboardInterrupt:
                print("\n已退出。")
                break
            except Exception as exc:
                print(f"执行异常: {exc}")
                self._speak("执行过程中遇到问题，请检查目标、相机或机械臂状态。")

    def _capture_image(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"视觉相机打开失败: camera_id={self.camera_id}")

        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise RuntimeError("视觉相机读取失败")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _release_shared_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _is_exit_command(self, text: str) -> bool:
        return any(word in text for word in ("退出", "结束程序", "关闭程序", "拜拜", "再见"))

    def _is_stop_tracking_command(self, text: str) -> bool:
        stop_words = ("停止", "停下", "结束", "别追", "不要追", "停止追踪", "停止跟随")
        tracking_words = ("追踪", "跟随", "跟着", "看着", "盯着", "tracking")
        return any(word in text for word in stop_words) and (
            self._tracking_is_running() or any(word in text for word in tracking_words)
        )

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

    def _format_nlu_error(self, parsed_data):
        if parsed_data.get("error") == "missing_target":
            return "我还不知道要操作哪个目标，请说清楚目标物体。"
        if parsed_data.get("error") == "target_not_in_coco":
            return "这个目标我暂时还不支持，请换成人、瓶子、杯子、椅子或手机等常见物体。"
        return "指令解析不完整，请重新说一遍。"

    def _speak(self, text: str):
        print(f"助手: {text}")
        self.tts.speak(text)

    def _reset_arm(self, arm, reason: str) -> None:
        print(f"准备执行{reason}，机械臂先回到 reset 位置")
        arm.reset()

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
                self.visual_tracking.release_cameras()
                self.tracking_thread = None
                return True
        if self.visual_tracking is not None:
            self.visual_tracking.release_cameras()
        self.tracking_thread = None
        return False

    def _reset_active_arm_if_needed(self, required: bool = False) -> bool:
        arm = None
        if self.robotic_interaction is not None:
            arm = self.robotic_interaction.arm
        elif self.visual_tracking is not None:
            arm = self.visual_tracking.arm

        if arm is None:
            if required:
                raise RuntimeError("需要机械臂 reset，但当前没有可用的机械臂控制器")
            return False

        self._reset_arm(arm, "下一步操作")
        return True

    def _reset_arm_before_intent(self, intent: str | None) -> None:
        if intent == "object_interaction":
            self._release_shared_camera()
            self._reset_arm(self._get_robotic_interaction().arm, "物体交互")
        elif intent == "visual_tracking":
            self._release_shared_camera()
            self._reset_arm(self._get_visual_tracking().arm, "目标追踪")
        else:
            self._reset_active_arm_if_needed(required=False)

    def _get_visual_tracking(self):
        if self.visual_tracking is None:
            from visual_tracking import VisualTracking

            self.visual_tracking = VisualTracking()
        return self.visual_tracking

    def _get_visual_qa(self):
        if self.visual_qa is None:
            from visual_qa import VisualQA

            self.visual_qa = VisualQA()
        return self.visual_qa

    def _get_robotic_interaction(self):
        if self.robotic_interaction is None:
            from robotic_interaction import RoboticInteraction

            self.robotic_interaction = RoboticInteraction()
        return self.robotic_interaction

    def _get_chatbot(self):
        if self.chatbot is None:
            from chat import ChatBot

            self.chatbot = ChatBot()
        return self.chatbot

    def _handle_tracking(self, target, current_image):
        if not target:
            return "请告诉我要追踪哪个目标。"
        self._stop_tracking_if_running()
        tracker = self._get_visual_tracking()
        self.tracking_thread = threading.Thread(
            target=tracker.track,
            args=(target,),
            daemon=True,
        )
        self.tracking_thread.start()
        return f"我开始追踪{target}了。"

    def _handle_vqa(self, user_text, current_image):
        return self._get_visual_qa().answer(current_image, user_text)

    def _handle_chat(self, user_text):
        return self._get_chatbot().reply(user_text)

    def _handle_interaction(self, target):
        if not target:
            return "请告诉我要操作哪个目标。"
        self._speak(f"好的，我开始寻找{target}，请注意机械臂移动。")
        robot = self._get_robotic_interaction()
        try:
            robot.interact(target)
        finally:
            robot.state = "IDLE"
        return f"我已经移动到{target}附近了，可以继续和我说话。"


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
