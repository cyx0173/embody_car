import re


class NLU:
    """意图识别 + 槽位填充模块"""

    TRACKING_WORDS = ("追踪", "跟踪", "盯着", "找到", "寻找", "看着", "锁定")
    ACTION_WORDS = ("抓", "拿", "夹", "指向", "移动到", "靠近", "复位", "回家", "停止", "刹车")
    VQA_WORDS = (
        "什么",
        "吗",
        "是不是",
        "有没有",
        "几",
        "多少",
        "描述",
        "看看",
        "看到",
        "画面",
        "屏幕",
        "电脑",
        "正在",
        "在做",
    )

    OBJECT_ALIASES = {
        "人": "person",
        "手机": "cell phone",
        "电脑": "laptop",
        "笔记本": "laptop",
        "键盘": "keyboard",
        "鼠标": "mouse",
        "杯子": "cup",
        "瓶子": "bottle",
        "书": "book",
        "椅子": "chair",
        "遥控器": "remote",
        "剪刀": "scissors",
        "盆栽": "potted plant",
    }

    def parse(self, text: str) -> dict:
        """
        解析用户文本，返回意图和槽位信息。
        返回格式: {"intent": str, "target_object": str, ...}
        """
        normalized = (text or "").strip().lower()
        target = self._extract_target(normalized)

        if any(word in normalized for word in self.ACTION_WORDS):
            return {"intent": "robotic_interaction", "target_object": target, "raw_text": text}
        if any(word in normalized for word in self.TRACKING_WORDS):
            return {"intent": "visual_tracking", "target_object": target, "raw_text": text}
        if any(word in normalized for word in self.VQA_WORDS) or target is None:
            return {"intent": "visual_qa", "target_object": target, "raw_text": text}
        return {"intent": "visual_qa", "target_object": target, "raw_text": text}

    def _extract_target(self, text: str) -> str | None:
        for cn_name, yolo_name in self.OBJECT_ALIASES.items():
            if cn_name in text:
                return yolo_name

        match = re.search(r"(?:track|find|follow|grab|pick|detect)\s+([a-zA-Z ]+)", text)
        if match:
            return match.group(1).strip()
        return None
