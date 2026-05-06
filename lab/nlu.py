class NLU:
    """意图识别 + 槽位填充模块"""

    def parse(self, text: str) -> dict:
        """
        解析用户文本，返回意图和槽位信息。
        返回格式: {"intent": str, "target_object": str, ...}
        """
        raise NotImplementedError("请实现意图识别逻辑")
