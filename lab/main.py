from asr import QwenASR as ASR
from nlu import NLU
from tts import TTS as TTS


class EmbodiedAgent:
    def __init__(self):
        self.asr = ASR()
        self.nlu = NLU().init()
        self.tts = TTS()
        self.visual_tracking = None
        self.visual_qa = None
        self.robotic_interaction = None

    def run(self):
        print("具身智能助手已启动，请下达指令...")
        while True:
            try:
                user_text = self.asr.listen()
                if not user_text:
                    continue

                print(f"用户: {user_text}")
                if self._is_exit_command(user_text):
                    self._speak("好的，已退出。")
                    break

                parsed_data = self.nlu.predict(user_text)
                print(f"NLU: {parsed_data}")
                if not parsed_data.get("valid", True):
                    self._speak(self._format_nlu_error(parsed_data))
                    continue

                intent = parsed_data.get("intent")
                target = parsed_data.get("target")

                current_image = self._capture_image()

                if intent == "visual_tracking":
                    reply = self._handle_tracking(target, current_image)
                    self._speak(reply)
                elif intent == "visual_understanding":
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
        return None

    def _is_exit_command(self, text: str) -> bool:
        return any(word in text for word in ("退出", "停止", "结束", "拜拜", "再见"))

    def _format_nlu_error(self, parsed_data):
        if parsed_data.get("error") == "missing_target":
            return "我还不知道要操作哪个目标，请说清楚目标物体。"
        if parsed_data.get("error") == "target_not_in_coco":
            return "这个目标我暂时还不支持，请换成人、瓶子、杯子、椅子或手机等常见物体。"
        return "指令解析不完整，请重新说一遍。"

    def _speak(self, text: str):
        print(f"助手: {text}")
        self.tts.speak(text)

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

    def _handle_tracking(self, target, current_image):
        tracker = self._get_visual_tracking()
        tracker.track(target)
        return "目标追踪完成"

    def _handle_vqa(self, user_text, current_image):
        return self._get_visual_qa().answer(current_image, user_text)

    def _handle_interaction(self, target):
        if not target:
            return "请告诉我要操作哪个目标。"
        self._speak(f"好的，我开始寻找{target}，请注意机械臂移动。")
        robot = self._get_robotic_interaction()
        robot.interact(target)
        return f"{target}交互完成"


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
