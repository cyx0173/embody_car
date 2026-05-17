from asr import QwenASR as ASR
from nlu import NLU
from tts import TTS as TTS
from visual_tracking import VisualTracking
from visual_qa import VisualQA
from robotic_interaction import RoboticInteraction


class EmbodiedAgent:
    def __init__(self):
        self.asr = ASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.visual_tracking = VisualTracking()
        self.visual_qa = VisualQA()
        self.robotic_interaction = RoboticInteraction()

    def run(self):
        print("具身智能助手已启动，请下达指令...")
        while True:
            try:
                user_text = self.asr.listen()
                if not user_text:
                    continue

                print(f"用户: {user_text}")

                parsed_data = self.nlu.predict(user_text)
                intent = parsed_data.get("intent")
                target = parsed_data.get("target")

                current_image = self._capture_image()

                if intent == "visual_tracking":
                    reply = self._handle_tracking(target, current_image)
                    self.tts.speak(reply)
                elif intent == "visual_understanding":
                    reply = self._handle_vqa(user_text, current_image)
                    if reply:
                        self.tts.speak(reply)
                elif intent == "object_interaction":
                    reply = self._handle_interaction(target)
                    self.tts.speak(reply)
                else:
                    self.tts.speak("未能识别有效意图，请重新下达指令。")
            except KeyboardInterrupt:
                print("\n已退出。")
                break

    def _capture_image(self):
        return None

    def _handle_tracking(self, target, current_image):
        self.visual_tracking.track(target, current_image)
        return "目标追踪完成"

    def _handle_vqa(self, user_text, current_image):
        return self.visual_qa.answer(current_image, user_text)

    def _handle_interaction(self, target):
        self.robotic_interaction.interact(target)
        return "交互完成"


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
