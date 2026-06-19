from __future__ import annotations

from vlm import ZHIPUAI_API_KEY, VLM_IMPORT_ERROR, ZhipuAI


CHAT_MODEL = "glm-4-flash"
MAX_HISTORY_MESSAGES = 8


class ChatBot:
    def __init__(self, api_key: str = ZHIPUAI_API_KEY, model: str = CHAT_MODEL):
        if VLM_IMPORT_ERROR is not None or ZhipuAI is None:
            raise RuntimeError(f"自然聊天依赖不可用，请先安装 zhipuai: {VLM_IMPORT_ERROR}")
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.history: list[dict[str, str]] = []

    def reply(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个温和、简洁的具身机器人助手。"
                    "用户如果只是闲聊，就自然对话；"
                    "如果用户想让机器人看、跟随或碰物体，提醒他可以直接下达任务指令。"
                    "回答使用中文，尽量控制在两三句话内。"
                ),
            },
            *self.history[-MAX_HISTORY_MESSAGES:],
            {"role": "user", "content": text},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = response.choices[0].message.content.strip()
        self.history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer
