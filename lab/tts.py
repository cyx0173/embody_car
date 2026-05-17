import re
import shutil

import numpy as np

try:
    import torch
    import sounddevice as sd
    from qwen_tts import Qwen3TTSModel
except ImportError as exc:
    torch = None
    sd = None
    Qwen3TTSModel = None
    TTS_IMPORT_ERROR = exc
else:
    TTS_IMPORT_ERROR = None


class TTS:
    def __init__(self, model_path="./Qwen3-TTS-0.6B-CustomVoice"):
        self.model = None
        self.sampling_rate = 24000
        self.enabled = False

        if TTS_IMPORT_ERROR is not None:
            print(f"[TTS] 语音合成依赖不可用，改为只打印文本: {TTS_IMPORT_ERROR}")
            return

        if shutil.which("sox") is None:
            print("[TTS] 未检测到 sox，若语音播放失败请执行: brew install sox")

        try:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            sd.default.device = 2
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
            self.enabled = True
            print("Jarvis TTS ready")
        except Exception as exc:
            print(f"[TTS] 初始化失败，改为只打印文本: {exc}")

    def _split_text(self, text):
        sentences = re.split(r'([。！？.!?;；])', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunks.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1 and sentences[-1]:
            chunks.append(sentences[-1])
        return [c for c in chunks if c.strip()]

    def speak(self, text, speaker="Vivian"):
        if not self.enabled or self.model is None:
            print(f"[TTS disabled] {text}")
            return

        chunks = self._split_text(text)
        for i, chunk in enumerate(chunks):
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language="Auto",
                )
                if wavs is not None and len(wavs) > 0:
                    audio_data = wavs[0].flatten().astype(np.float32)
                    sd.play(audio_data, self.sampling_rate)
                    sd.wait()
            except Exception as exc:
                print(f"[TTS] 播放失败，改为只打印文本: {exc}")
                print(f"[TTS disabled] {chunk}")
                self.enabled = False
                return


if __name__ == "__main__":
    tts = TTS()
    tts.speak("你好，tts正常")
