import os
import re
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/embody_car_numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

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


BASE_DIR = Path(__file__).resolve().parent
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
TTS_PUNCT_RE = re.compile(r"[，,、：:（）()\[\]【】《》<>“”\"'`*_#~|\\/@]+")
SPACE_RE = re.compile(r"\s+")


class TTS:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.sampling_rate = 24000
        self.enabled = False
        self.default_speaker = os.environ.get("TTS_SPEAKER", "vivian")
        self.default_language = os.environ.get("TTS_LANGUAGE", "chinese")
        self.do_sample = os.environ.get("TTS_DO_SAMPLE", "0").lower() in ("1", "true", "yes")
        self.temperature = float(os.environ.get("TTS_TEMPERATURE", "0.6"))
        if model_path is None:
            model_path = str(BASE_DIR / "Qwen3-TTS-0.6B-CustomVoice")

        if TTS_IMPORT_ERROR is not None:
            print(f"[TTS] 语音合成依赖不可用，改为只打印文本: {TTS_IMPORT_ERROR}")
            return

        if shutil.which("sox") is None:
            print("[TTS] 未检测到 sox，若语音播放失败请执行: brew install sox")

        try:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            output_device = os.environ.get("TTS_OUTPUT_DEVICE")
            if output_device:
                sd.default.device = (None, int(output_device))
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
        return [
            chunk
            for sentence in SENTENCE_SPLIT_RE.split(str(text))
            if (chunk := self._normalize_for_tts(sentence))
        ]

    def _normalize_for_tts(self, text):
        text = TTS_PUNCT_RE.sub(" ", str(text))
        text = text.replace("...", " ").replace("…", " ")
        text = text.replace("-", " ").replace("—", " ")
        return SPACE_RE.sub(" ", text).strip()

    def speak(self, text, speaker=None, language=None):
        if not self.enabled or self.model is None:
            print(f"[TTS disabled] {text}")
            return

        speaker = speaker or self.default_speaker
        language = language or self.default_language
        chunks = self._split_text(text)
        for i, chunk in enumerate(chunks):
            try:
                gen_kwargs = {
                    "do_sample": self.do_sample,
                    "subtalker_dosample": self.do_sample,
                }
                if self.do_sample:
                    gen_kwargs.update(
                        {
                            "temperature": self.temperature,
                            "top_k": 20,
                            "top_p": 0.8,
                            "subtalker_temperature": self.temperature,
                            "subtalker_top_k": 20,
                            "subtalker_top_p": 0.8,
                        }
                    )
                wavs, sr = self.model.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                    **gen_kwargs,
                )
                if wavs is not None and len(wavs) > 0:
                    audio_data = wavs[0].flatten().astype(np.float32)
                    sd.play(audio_data, sr)
                    sd.wait()
            except Exception as exc:
                print(f"[TTS] 播放失败，改为只打印文本: {exc}")
                print(f"[TTS disabled] {chunk}")
                self.enabled = False
                return


if __name__ == "__main__":
    tts = TTS()
    tts.speak("你好，欢迎使用Jarvis语音合成功能！这是一个测试。")
