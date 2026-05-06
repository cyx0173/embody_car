import os
import joblib
import librosa
import requests
import numpy as np
import sounddevice as sd
import sherpa_onnx
from typing import Union
from sentence_transformers import SentenceTransformer
from sherpa_onnx import OfflineRecognizer, VadModelConfig, SileroVadModelConfig, VoiceActivityDetector

# ══════════════════════════════════════════════════════════
#  1. 路径与全局配置
# ══════════════════════════════════════════════════════════
ASR_PATH = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
VAD_PATH = 'model/VAD'
MLP_PATH = 'model/MLP/command_classifier.pkl'
CONTROL_URL = "http://192.168.192.123:5000/control"  
SAMPLE_RATE = 16000

# ══════════════════════════════════════════════════════════
#  2. 模型单例加载 (确保只加载一次)
# ══════════════════════════════════════════════════════════
print('📦 正在加载语言模型与分类头...')
# 加载 MiniLM 语义向量模型 (BERT 家族)
lm_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# 加载你训练好的 MLP 逻辑分类器
mlp_model = joblib.load(MLP_PATH)
print('✅ 本地推理模型加载完成')

# ══════════════════════════════════════════════════════════
#  3. 核心功能封装：意图识别函数
# ══════════════════════════════════════════════════════════
def get_intent(text: str) -> str:
    """
    外部调用接口：将文本转换为小车指令标签
    cmd = get_intent("小车请左转五秒")
    """
    if not text or len(text.strip()) == 0:
        return "无操作"
    
    # 1. 特征提取 (Embedding)
    embedding = lm_model.encode(text).reshape(1, -1)
    
    # 2. 分类预测
    prediction = mlp_model.predict(embedding)[0]
    
    return prediction

# ══════════════════════════════════════════════════════════
#  4. 辅助类与函数
# ══════════════════════════════════════════════════════════
class Paraformer:
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 4):
        self._recognizer = OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider='cpu',
        )

    def transcribe(self, audio: np.ndarray, sample_rate=16000) -> str:
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text

def send_command_to_car(command_label: str):
    """根据标签发送 HTTP 请求控制小车硬件"""
    # 映射标签到硬件指令（根据你的硬件接口调整）
    mapping = {
        "前进": "FORWARD",
        "后退": "BACKWARD",
        "左转": "LEFT",
        "右转": "RIGHT",
        "停止": "STOP"
    }
    
    cmd_value = mapping.get(command_label, "STOP")
    
    try:
        response = requests.post(CONTROL_URL, json={'command': cmd_value}, timeout=0.5)
        if response.status_code == 200:
            print(f"🚀 硬件执行成功: {cmd_value}")
        else:
            print(f"⚠️ 硬件响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到小车控制服务器: {e}")

# ══════════════════════════════════════════════════════════
#  5. 主程序运行逻辑
# ══════════════════════════════════════════════════════════
def main():
    # 初始化 ASR
    print('🎙️ 正在加载 ASR 语音识别模型...')
    asr = Paraformer(
        model_path=f'{ASR_PATH}/model.int8.onnx',
        tokens_path=f'{ASR_PATH}/tokens.txt'
    )

    # 初始化 VAD
    vad_config = VadModelConfig(
        SileroVadModelConfig(model=f'{vad_path}/silero_vad.onnx', min_silence_duration=0.25),
        sample_rate=SAMPLE_RATE
    )
    window_size = vad_config.silero_vad.window_size
    vad = VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
    
    samples_per_read = int(0.1 * SAMPLE_RATE) # 每100ms采样一次
    audio_buffer = []
    idx = 1

    print('\n🎧 系统就绪，请说话... (Ctrl+C 退出)')
    
    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as stream:
            while True:
                # 读取音频流
                samples, _ = stream.read(samples_per_read)
                samples = samples.reshape(-1)

                # VAD 状态更新
                audio_buffer = np.concatenate([audio_buffer, samples])
                while len(audio_buffer) > window_size:
                    vad.accept_waveform(audio_buffer[:window_size])
                    audio_buffer = audio_buffer[window_size:]

                # 当 VAD 检测到一句话结束时
                while not vad.empty():
                    # 1. 语音转文字
                    raw_text = asr.transcribe(vad.front.samples, SAMPLE_RATE)
                    vad.pop()
                    
                    if raw_text.strip():
                        print(f"\n[第 {idx} 句识别]: {raw_text}")
                        
                        # 2. 调用封装好的意图识别接口
                        cmd = get_intent(raw_text)
                        
                        # 3. 执行动作
                        if cmd == '无操作':
                            print("🤷 未识别到有效意图")
                        else:
                            print(f"🎯 识别到指令标签: 【{cmd}】")
                            send_command_to_car(cmd)
                        
                        idx += 1
                        
    except KeyboardInterrupt:
        print('\n🛑 识别已手动结束')
    except Exception as e:
        print(f"🔥 程序发生异常: {e}")

if __name__ == "__main__":
    # 兼容之前脚本定义的路径变量
    vad_path = VAD_PATH 
    main()