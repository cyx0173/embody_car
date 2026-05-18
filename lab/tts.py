from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/embody_car_numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

try:
    import sounddevice as sd
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from misaki.zh import ZHG2P
except ImportError as exc:
    sd = None
    sf = None
    Kokoro = None
    ZHG2P = None
    TTS_IMPORT_ERROR = exc
else:
    TTS_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent

KOKORO_MODEL_PATH = Path(
    os.environ.get(
        "KOKORO_MODEL_PATH",
        "/Users/kismet/kokoro_test/kokoro-v1.0-opset19-mixed.onnx",
    )
)

KOKORO_VOICES_PATH = Path(
    os.environ.get(
        "KOKORO_VOICES_PATH",
        "/Users/kismet/kokoro_test/voices-v1.0.bin",
    )
)

SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
SPACE_RE = re.compile(r"\s+")

# 这里只清理明显不适合直接读的符号，不要把中文逗号等全部删掉太狠
TTS_PUNCT_RE = re.compile(r"[（）()\[\]【】《》<>“”\"'`*_#~|\\/@]+")


class TTS:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.g2p = None
        self.enabled = False

        self.default_language = os.environ.get("TTS_LANGUAGE", "zh")
        self.default_speaker = os.environ.get("TTS_SPEAKER", "zm_yunxi")
        self.speed = float(os.environ.get("TTS_SPEED", "1.0"))
        self.save_debug = os.environ.get("TTS_SAVE_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
        )

        if TTS_IMPORT_ERROR is not None:
            print(f"[TTS] Kokoro 依赖不可用，改为只打印文本: {TTS_IMPORT_ERROR}")
            return

        model_file = Path(model_path) if model_path is not None else KOKORO_MODEL_PATH
        voices_file = KOKORO_VOICES_PATH

        if not model_file.exists():
            print(f"[TTS] Kokoro 模型不存在: {model_file}")
            return

        if not voices_file.exists():
            print(f"[TTS] Kokoro voices 文件不存在: {voices_file}")
            return

        try:
            output_device = os.environ.get("TTS_OUTPUT_DEVICE")
            if output_device:
                sd.default.device = (None, int(output_device))

            print("[TTS] sounddevice default device:", sd.default.device)

            self.model = Kokoro(str(model_file), str(voices_file))
            self.g2p = ZHG2P()
            self.enabled = True

            print(
                f"Kokoro TTS ready: model={model_file}, "
                f"voice={self.default_speaker}, lang={self.default_language}"
            )

        except Exception as exc:
            print(f"[TTS] Kokoro 初始化失败，改为只打印文本: {type(exc).__name__}: {exc}")

    def _split_text(self, text: str) -> list[str]:
        return [
            chunk
            for sentence in SENTENCE_SPLIT_RE.split(str(text))
            if (chunk := self._normalize_for_tts(sentence))
        ]

    def _normalize_for_tts(self, text: str) -> str:
        text = str(text)
        text = TTS_PUNCT_RE.sub(" ", text)
        text = text.replace("...", " ").replace("…", " ")
        text = text.replace("-", " ").replace("—", " ")
        return SPACE_RE.sub(" ", text).strip()

    def _is_chinese_language(self, language: str) -> bool:
        language = language.lower()
        return language in ("zh", "cn", "cmn", "chinese", "mandarin", "zh-cn")

    def speak(self, text: str, speaker: str | None = None, language: str | None = None):
        if not self.enabled or self.model is None:
            print(f"[TTS disabled] {text}")
            return

        speaker = speaker or self.default_speaker
        language = language or self.default_language

        chunks = self._split_text(text)
        if not chunks:
            return

        for chunk in chunks:
            try:
                print(f"[TTS] 开始生成: {chunk}", flush=True)

                if self._is_chinese_language(language):
                    if self.g2p is None:
                        print("[TTS] 中文 G2P 不可用")
                        print(f"[TTS disabled] {chunk}")
                        return

                    phonemes, _ = self.g2p(chunk)
                    print(f"[TTS] 音素: {phonemes}", flush=True)

                    samples, sample_rate = self.model.create(
                        phonemes,
                        voice=speaker,
                        speed=self.speed,
                        lang="zh",
                        is_phonemes=True,
                    )
                else:
                    samples, sample_rate = self.model.create(
                        chunk,
                        voice=speaker,
                        speed=self.speed,
                        lang=language,
                    )

                audio_data = np.asarray(samples, dtype=np.float32).flatten()

                if audio_data.size == 0:
                    print("[TTS] 生成音频为空")
                    continue

                abs_max = float(np.max(np.abs(audio_data)))
                print(
                    f"[TTS] sample_rate={sample_rate}, "
                    f"shape={audio_data.shape}, abs_max={abs_max:.6f}",
                    flush=True,
                )

                if abs_max < 1e-6:
                    print("[TTS] 生成音频几乎全是 0")
                    continue

                audio_data = audio_data / abs_max * 0.8

                if self.save_debug:
                    out_path = BASE_DIR / "tts_debug.wav"
                    sf.write(str(out_path), audio_data, sample_rate)
                    print(f"[TTS] 已保存: {out_path}")

                sd.play(audio_data, sample_rate)
                sd.wait()
                print("[TTS] 播放完成", flush=True)

            except Exception as exc:
                print(f"[TTS] Kokoro 播放失败，改为只打印文本: {type(exc).__name__}: {exc}")
                print(f"[TTS disabled] {chunk}")
                return


if __name__ == "__main__":
    tts = TTS()
    tts.speak("你好，这是一个语音合成测试，听起来怎么样？")