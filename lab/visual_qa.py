from vlm import VLM
class VisualQA:
    def __init__(self):
        self.vlm = VLM()

    def answer(self, image, question: str, context: str | None = None) -> str:
        prompt = question
        if context:
            prompt = (
                "你是一个面向机械臂任务的视觉助手。请结合图像和下面的YOLO检测结果回答用户，"
                "如果用户问正在做什么，请优先描述屏幕/桌面中可见的活动；回答简短自然。\n"
                f"{context}\n"
                f"用户问题: {question}"
            )
        return self.vlm.generate(image, prompt)
