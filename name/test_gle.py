import base64
from openai import OpenAI

# 任务二：调用 VLM API 配置 [cite: 17, 21]
client = OpenAI(
    api_key="你的_API_KEY_粘贴在这里", # [cite: 72]
    base_url="83c7f6e5a6a246de80603667e6d6f33a.RDlij4U5CFb5622m" # [cite: 57, 73]
)

def encode_image(image_path):
    """将本地图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_arm_action(image_path):
    base64_image = encode_image(image_path)
    
    # 构建 Data URL [cite: 90]
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
    # 任务二：发起 API 请求 [cite: 81]
    completion = client.chat.completions.create(
        model="glm-4.6v-flash", # 使用免费模型 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述图片中人物手臂的动作，并给出对应的机械臂控制建议。"}, # [cite: 92]
                    {"type": "image_url", "image_url": {"url": data_url}} # [cite: 90]
                ]
            }
        ],
        extra_body={"enable_thinking": False} # 禁用思考以加快响应 
    )
    
    # 解析并打印模型输出 [cite: 106, 110]
    content = completion.choices[0].message.content
    print("模型输出内容:", content)
    return content