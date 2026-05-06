from asr import QwenASR as ASR
from nlu import NLU
from tts import TTS
from visual_tracking import VisualTracking
from visual_qa import VisualQA
from robotic_interaction import RoboticInteraction


class EmbodiedAgent:
    def __init__(self):
        self.asr = ASR()
        self.nlu = NLU()
        self.tts = TTS()
        self.visual_tracking = VisualTracking()
        self.visual_qa = VisualQA()
        self.robotic_interaction = RoboticInteraction()

    def run(self):
        print("具身智能助手已启动，请下达指令...")
        while True:
            try:
                # 1. 语音识别
                user_text = self.asr.listen()
                if not user_text:
                    continue

                print(f"用户: {user_text}")

                # 2. 意图识别 + 槽位填充
                parsed_data = self.nlu.parse(user_text)
                intent = parsed_data.get("intent")
                target = parsed_data.get("target_object")

                # 3. 获取当前图像
                current_image = self._capture_image()

                # 4. 意图路由分发
                if intent == "visual_tracking":
                    self._handle_tracking(target, current_image)
                elif intent == "visual_qa":
                    self._handle_vqa(user_text, current_image)
                elif intent == "robotic_interaction":
                    self._handle_interaction(target, current_image)
                else:
                    print("未能识别有效意图，请重新下达指令。")

            except KeyboardInterrupt:
                print("\n已退出。")
                break

    def _capture_image(self):
        """获取当前视觉输入（后续替换为实际摄像头/图像源）"""
        # TODO: 替换为实际图像采集逻辑
        return None

    def _handle_tracking(self, target, current_image):
        """视觉追踪处理"""
        raise NotImplementedError("请实现视觉追踪逻辑")

    def _handle_vqa(self, user_text, current_image):
        if current_image is None:
            print("[错误] 图像为空，请检查摄像头或图像采集逻辑。")
            return
        answer = self.visual_qa.answer(current_image, user_text)
        self.tts.speak(answer)

    def _handle_interaction(self, target, current_image):
        """机器人交互处理"""
        raise NotImplementedError("请实现机器人交互逻辑")


if __name__ == "__main__":
    agent = EmbodiedAgent()
    agent.run()
