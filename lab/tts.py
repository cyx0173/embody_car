import subprocess
from pathlib import Path

import torch
import sounddevice as sd
import numpy as np
import re


class TTS:
    def __init__(self, model_path="./Qwen3-TTS-0.6B-CustomVoice"):
        self.model = None
        self.fallback_say = False
        self.sampling_rate = 24000
        if not Path(model_path).exists():
            print(f"[TTS] 未找到本地模型 {model_path}，将使用系统say命令。")
            self.fallback_say = True
            return

        from qwen_tts import Qwen3TTSModel

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        sd.default.device = 2

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        print("Jarvis TTS ready")

    def _split_text(self, text):
        sentences = re.split(r'([。！？.!?;；])', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunks.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1 and sentences[-1]:
            chunks.append(sentences[-1])
        return [c for c in chunks if c.strip()]

    def speak(self, text, speaker="Vivian"):
        if self.fallback_say:
            print(f"助手: {text}")
            subprocess.run(["say", text], check=False)
            return

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
    tts.speak("你好，tts正常")
