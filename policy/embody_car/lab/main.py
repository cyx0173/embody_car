from __future__ import annotations

from asr import QwenASR
from nlu import NLU
from tts import TTS
from scene_understanding import SceneUnderstanding
from pick_place import PickPlaceSkill
from support.chat import ChatBot
from support.runtime import RobotRuntime, format_nlu_error, target_bowl_from_nlu


class VoiceRobotApp:
    def __init__(self):
        self.robot = RobotRuntime()
        self.scene_understanding = SceneUnderstanding()
        self.pick_place = PickPlaceSkill()

        self.asr = QwenASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.chatbot = ChatBot()

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
                    self._speak(format_nlu_error(parsed))
                    continue

                intent = parsed.get("intent")
                target = parsed.get("target")
                self.robot.stop_tracking_if_running()

                if intent == "visual_tracking":
                    reply = self._handle_object_tracking(target)
                    self._speak(reply)
                elif intent == "visual_understanding":
                    current_image = self.robot.capture_scene_image()
                    reply = self._handle_scene_understanding(user_text, current_image)
                    if reply:
                        self._speak(reply)
                elif intent == "object_interaction":
                    reply = self._handle_object_interaction(target)
                    self._speak(reply)
                elif intent == "object_grasp":
                    reply = self._handle_pick_place(parsed)
                    self._speak(reply)
                elif intent == "voice_chat":
                    reply = self._handle_chat(user_text)
                    self._speak(reply)
                elif intent == "reset_arm":
                    self.robot.reset_arm()
                    self._speak("好的，已重置机械臂。")
                else:
                    self._speak("未能识别有效意图，请重新下达指令。")
            except KeyboardInterrupt:
                print("\n已退出。")
                break
            except Exception as exc:
                print(f"执行异常: {exc}")
                self._speak("执行过程中遇到问题，请检查目标、相机或机械臂状态。")

    def _speak(self, text: str) -> None:
        print(f"助手: {text}")
        self.tts.speak(text)

    def _handle_object_tracking(self, target: str | None) -> str | None:
        if not target:
            return "请告诉我要追踪哪个目标。"
        self.robot.start_tracking(target)
        return f"我开始追踪{target}了。"

    def _handle_scene_understanding(self, user_text: str, current_image) -> str | None:
        return self.scene_understanding.answer(current_image, user_text)

    def _handle_chat(self, user_text: str) -> str | None:
        return self.chatbot.reply(user_text)

    def _handle_object_interaction(self, target: str | None) -> str | None:
        if not target:
            return "请告诉我要操作哪个目标。"
        self._speak(f"好的，我开始寻找{target}，请注意机械臂移动。")
        try:
            self.robot.object_interaction.interact(target)
        finally:
            self.robot.object_interaction.state = "IDLE"
        return f"我已经移动到{target}附近了，可以继续和我说话。"

    def _handle_pick_place(self, parsed: dict) -> str | None:
        target = parsed.get("target")
        if not target:
            return "请告诉我要抓取哪个目标。"

        target_bowl = target_bowl_from_nlu(parsed)
        self._speak(f"好的，我开始抓取{target}，准备放到{target_bowl}边的碗里。")
        self.robot.release_for_pick_place()
        try:
            return self.pick_place.run(target=target, target_bowl=target_bowl)
        finally:
            self.robot.restore_after_pick_place()


if __name__ == "__main__":
    app = VoiceRobotApp()
    app.run()
