import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import sounddevice as sd
import numpy as np
import re
from qwen_tts import Qwen3TTSModel


class TTS:
    def __init__(self, model_path="./Qwen3-TTS-0.6B-CustomVoice"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        #sd.default.device = 2

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        print(self.device)
        self.sampling_rate = 24000

    def _split_text(self, text):
        sentences = re.split(r'([。！？.!?;；])', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunks.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1 and sentences[-1]:
            chunks.append(sentences[-1])
        return [c for c in chunks if c.strip()]

    def speak(self, text, speaker="Vivian"):
        chunks = self._split_text(text)
        for i, chunk in enumerate(chunks):
            wavs, sr = self.model.generate_custom_voice(
                text=chunk,
                speaker=speaker,
                language="Auto",
            )
            if wavs is not None and len(wavs) > 0:
                audio_data = wavs[0].flatten().astype(np.float32)
                sd.play(audio_data, self.sampling_rate)
                sd.wait()


if __name__ == "__main__":
    tts = TTS()
    while True:
        try:
            user_input = input("请输入文本: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出程序。")
                break
                
            if not user_input:
                continue
                
            tts.speak(user_input)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")
