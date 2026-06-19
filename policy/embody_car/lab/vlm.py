import base64
import cv2
import numpy as np
try:
    from zhipuai import ZhipuAI
except ImportError as exc:
    ZhipuAI = None
    VLM_IMPORT_ERROR = exc
else:
    VLM_IMPORT_ERROR = None

ZHIPUAI_API_KEY = "8347ecf1ad7f45ec92164105c33773bc.wivz1tEE2xTWOUbc"

class VLM:
    def __init__(self, api_key: str = ZHIPUAI_API_KEY):
        if VLM_IMPORT_ERROR is not None or ZhipuAI is None:
            raise RuntimeError(f"视觉语义理解依赖不可用，请先安装 zhipuai: {VLM_IMPORT_ERROR}")
        self.client = ZhipuAI(api_key=api_key)

    def _rgb_to_b64(self, rgb: np.ndarray) -> str:
        if rgb.dtype == np.uint8:
            u8 = rgb
        else:
            u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return base64.b64encode(cv2.imencode(".jpg", u8)[1]).decode()

    def generate(self, image, text: str) -> str:
        if image is None:
            raise ValueError("视觉语义理解需要一帧图像，但当前 image=None")
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


if __name__ == "__main__":
    vlm = VLM()

    # Test 1: path string
    result1 = vlm.generate("vlm_test.png", "描述这张图片")
    print("Test 1 (path):", result1)

    # Test 2: numpy array (random RGB image)
    img = np.random.rand(224, 224, 3).astype(np.float32)
    result2 = vlm.generate(img, "这张图里有什么？")
    #这个应该是正常的调用方式
    print("Test 2 (numpy):", result2)
'''
test result:
(embody) chengyx@chengyxdeMacBook-Air lab % python vlm.py
Test 1 (path): 这张图片展示了一只小狗和一只小猫并排坐着的温馨画面。  
- 小狗看起来像是一只幼年的金毛寻回犬，有着棕色和白色的皮毛，大而圆的眼睛显得非常友善。它的耳朵耷拉着，给人一种温顺的感觉。
- 小猫则是一只白色长毛品种，可能是美国短毛猫或类似的品种。它有一双明亮的蓝色眼睛，鼻子是粉红色的。它的耳朵形状独特，内侧是粉色的。

它们都望向镜头，似乎在微笑着打招呼。整个画面背景为纯白色，使得主题更加突出。
Test 2 (numpy): 这是一张白噪声的图片，展示了一种复杂的视觉效果，类似于电视屏幕无信号时出现的静态。它由成千上万的小色点组成，这些色点似乎是随机分布的，并覆盖了整个图像区域。这些色点涵盖了多种颜色，看起来没有特定的模式或顺序。
(embody) chengyx@chengyxdeMacBook-Air lab % 
'''
