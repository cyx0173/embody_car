import base64
import cv2
import numpy as np
from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "8347ecf1ad7f45ec92164105c33773bc.wivz1tEE2xTWOUbc"

class VLM:
    def __init__(self, api_key: str = ZHIPUAI_API_KEY):
        self.client = ZhipuAI(api_key=api_key)

    def _rgb_to_b64(self, rgb: np.ndarray) -> str:
        u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return base64.b64encode(cv2.imencode(".jpg", u8)[1]).decode()

    def generate(self, image, text: str) -> str:
        if isinstance(image, str):
            b64_str = base64.b64encode(open(image, "rb").read()).decode()
        else:
            b64_str = self._rgb_to_b64(image)

        response = self.client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
