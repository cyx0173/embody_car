import json
from openai import OpenAI

# 1. API 配置
API_KEY = "sk-661cca974e7a4b4ea968b324ef8669dc" 
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 2. 定义系统提示词 (这是控制小车的“大脑”逻辑)
SYSTEM_PROMPT = """
你是一个高度智能的小车控制指令转换器。你的任务是将用户的自然语言转换为机器可执行的指令。

### 允许的指令集:
- MOVE_FORWARD(speed, duration): 前进。speed(0-100), duration(秒)
- MOVE_BACKWARD(speed, duration): 后退。
- TURN_LEFT(speed, duration): 向左转动或侧移。
- TURN_RIGHT(speed, duration): 向右转动或侧移。
- STOP(): 停止。

### 转换逻辑逻辑（非常重要）:
1. 忽略“小车”、“想”、“给我”、“请”等废话。
2. 提取【动作】、【速度】和【时间/角度】。
3. 如果用户只说了时间（如“五秒”），没有说速度，默认速度设为 50。
4. 如果用户说“向左/向右移动”，统一映射到 TURN_LEFT 或 TURN_RIGHT。
5. 必须只输出指令本身，不要有任何解释。多条指令用分号隔开。

### 示例:
- "小车想左侧移动五秒" -> TURN_LEFT(50, 5)
- "快点向后退" -> MOVE_BACKWARD(80, 2)
- "停下" -> STOP()
- "向右转并在3秒后停止" -> TURN_RIGHT(50, 3);STOP()
"""
def get_car_commands(user_voice_text):
    """
    将语音识别出的文本转换为小车指令
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_voice_text}
            ],
            temperature=0.1, # 降低随机性，保证指令稳定
            stream=False
        )
        
        command_text = response.choices[0].message.content.strip()
        return command_text
    except Exception as e:
        return f"ERROR: {str(e)}"

# 3. 模拟控制循环
if __name__ == "__main__":
    print("=== 小车语音指令解析系统 (DeepSeek 版) ===")
    print("输入 'quit' 退出")
    
    while True:
        # 这里假设你已经有了语音转文字的结果
        # 在实际项目中，这里应该是调用麦克风识别后的文本
        user_input = input("\n[语音识别文本]: ")
        
        if user_input.lower() == 'quit':
            break
            
        if not user_input:
            continue
            
        print("正在解析指令...")
        commands = get_car_commands(user_input)
        
        # 这里就是你需要的标准化控制文本
        print(f"解析结果: \033[32m{commands}\033[0m") 
